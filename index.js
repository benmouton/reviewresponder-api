var express = require('express');
var cors = require('cors');
var OpenAI = require('openai');
var multer = require('multer');
var fs = require('fs');
var path = require('path');
var { Pool } = require('pg');
var rateLimit = require('express-rate-limit');

var app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// ── Health check endpoint ───────────────────────────────────
app.get('/health', async function(req, res) {
  var dbOk = false;
  try {
    await pool.query('SELECT 1');
    dbOk = true;
  } catch (e) {
    // db unreachable
  }
  var status = dbOk ? 200 : 503;
  res.status(status).json({
    status: dbOk ? 'ok' : 'degraded',
    timestamp: new Date().toISOString(),
    db: dbOk ? 'connected' : 'unreachable',
    uptime: process.uptime(),
  });
});

// Rate limiting: 30 requests per minute per IP for generation endpoints
var generateLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 30,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests, please try again in a minute.' },
});
app.use('/generate', generateLimiter);
app.use('/ocr', generateLimiter);
app.use('/analyze-review', generateLimiter);
app.use('/check-limit', generateLimiter);

// Stricter rate limit for admin endpoint (10 per minute)
var adminLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 10,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests.' },
});
app.use('/admin', adminLimiter);

var upload = multer({
  dest: 'uploads/',
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB max
  fileFilter: function(req, file, cb) {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'));
    }
  },
});

var openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// ── Server-side Generation Counter (Postgres) ───────────────
var pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
});

async function initDatabase() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS generations (
      device_id TEXT NOT NULL,
      month TEXT NOT NULL,
      count INTEGER NOT NULL DEFAULT 0,
      cost_micro_usd BIGINT NOT NULL DEFAULT 0,
      PRIMARY KEY (device_id, month)
    )
  `);
  // Add cost column to existing tables that were created without it
  await pool.query(`
    ALTER TABLE generations ADD COLUMN IF NOT EXISTS cost_micro_usd BIGINT NOT NULL DEFAULT 0
  `);
  console.log('[DB] Generations table ready');
}

initDatabase().catch(function(err) {
  console.error('[DB] Failed to initialize:', err.message);
});

var FREE_GENERATION_LIMIT = 15;

function getCurrentMonth() {
  var now = new Date();
  return now.getFullYear() + '-' + String(now.getMonth() + 1).padStart(2, '0');
}

async function getGenerationCount(deviceId) {
  var month = getCurrentMonth();
  var result = await pool.query(
    'SELECT count FROM generations WHERE device_id = $1 AND month = $2',
    [deviceId, month]
  );
  return result.rows.length > 0 ? result.rows[0].count : 0;
}

// Cost constants (micro-USD = millionths of a dollar) for gpt-4o-mini
// $0.15/1M input tokens, $0.60/1M output tokens
var GPT4O_MINI_INPUT_MICRO_USD_PER_TOKEN = 0.15;   // $0.15 per 1M = 0.15 micro-USD per token
var GPT4O_MINI_OUTPUT_MICRO_USD_PER_TOKEN = 0.60;  // $0.60 per 1M = 0.60 micro-USD per token

function calcCostMicroUsd(inputTokens, outputTokens) {
  return Math.round(
    (inputTokens * GPT4O_MINI_INPUT_MICRO_USD_PER_TOKEN) +
    (outputTokens * GPT4O_MINI_OUTPUT_MICRO_USD_PER_TOKEN)
  );
}

async function incrementGeneration(deviceId, costMicroUsd) {
  var month = getCurrentMonth();
  await pool.query(`
    INSERT INTO generations (device_id, month, count, cost_micro_usd) VALUES ($1, $2, 1, $3)
    ON CONFLICT (device_id, month) DO UPDATE
      SET count = generations.count + 1,
          cost_micro_usd = generations.cost_micro_usd + $3
  `, [deviceId, month, costMicroUsd || 0]);
  return await getGenerationCount(deviceId);
}

// ── Check generation limit endpoint ─────────────────────────
app.post('/check-limit', async function(req, res) {
  var deviceId = req.body.deviceId;
  var isPro = req.body.isPro || false;

  if (!deviceId) {
    return res.status(400).json({ error: 'deviceId is required' });
  }

  try {
    var count = await getGenerationCount(deviceId);
    res.json({
      count: count,
      limit: FREE_GENERATION_LIMIT,
      remaining: isPro ? 999999 : Math.max(0, FREE_GENERATION_LIMIT - count),
      isAtLimit: !isPro && count >= FREE_GENERATION_LIMIT,
    });
  } catch (err) {
    console.error('[check-limit] Error:', err.message);
    res.status(500).json({ error: 'Failed to check limit' });
  }
});

// NEW: Text-based Review Analysis Endpoint (used with on-device Apple Vision OCR)
// Receives plain text extracted on-device and uses GPT-4o-mini (much cheaper than sending images to GPT-4o)
app.post('/analyze-review', async function(req, res) {
  console.log('[Analyze] Request received');
  try {
    var text = req.body.text;

    if (!text || text.trim().length === 0) {
      return res.status(400).json({ error: 'No text provided' });
    }

    var completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: 'You analyze customer review text extracted via OCR from review platform screenshots (Google, Yelp, TripAdvisor). The text may include UI elements, reviewer info, and the actual review. Your job is to parse all of it. Respond with ONLY a JSON object, no other text.',
        },
        {
          role: 'user',
          content: 'Analyze this OCR-extracted text from a review screenshot. Respond with ONLY this JSON format:\n\n{"cleanedText": "just the actual review text, stripped of UI elements and metadata", "sentiment": "positive" or "negative" or "mixed", "reviewerName": "reviewer name if found, or null", "reviewerBadge": "badge like Elite 24 or Local Guide if found, or null", "photoCount": number of photos if mentioned or null, "checkInCount": number of check-ins if mentioned or null, "platform": "Google" or "Yelp" or "TripAdvisor" or null, "suggestions": ["array of specific improvement suggestions mentioned in the review, like faster service or better parking"]}\n\nOCR text:\n' + text,
        },
      ],
      max_tokens: 500,
    });

    var rawResponse = completion.choices[0].message.content;
    console.log('[Analyze] Response received');

    try {
      var cleaned = rawResponse.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
      var parsed = JSON.parse(cleaned);
      res.json({
        cleanedText: parsed.cleanedText || text,
        sentiment: parsed.sentiment || 'mixed',
        reviewerName: parsed.reviewerName || null,
        reviewerBadge: parsed.reviewerBadge || null,
        photoCount: parsed.photoCount || null,
        checkInCount: parsed.checkInCount || null,
        platform: parsed.platform || null,
        suggestions: parsed.suggestions || [],
      });
    } catch (parseError) {
      console.warn('[Analyze] JSON parse failed, returning defaults');
      res.json({
        cleanedText: text,
        sentiment: 'mixed',
        reviewerName: null,
        reviewerBadge: null,
        photoCount: null,
        checkInCount: null,
        platform: null,
        suggestions: [],
      });
    }
  } catch (error) {
    console.error('[Analyze] Error:', error.message);
    res.status(500).json({ error: 'Review analysis failed' });
  }
});

// LEGACY: OCR Endpoint (kept as fallback - sends image to GPT-4o)
app.post('/ocr', upload.single('image'), async function(req, res) {
  try {
    var imagePath = req.file.path;
    var imageData = fs.readFileSync(imagePath);
    var base64Image = imageData.toString('base64');
    var mimeType = req.file.mimetype || 'image/jpeg';

    var completion = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        {
          role: 'system',
          content: 'You are an OCR text extraction tool used in a business reputation management app. Your job is to read customer review screenshots and extract the review text plus reviewer metadata. This is a legitimate business tool that helps business owners respond professionally to customer feedback. Always extract and return the review text exactly as written.',
        },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'This is a screenshot of a customer review from a review platform like Google, Yelp, or TripAdvisor. Please respond in this exact JSON format:\n\n{"text": "the extracted review text here", "sentiment": "positive" or "negative" or "mixed", "reviewerName": "name if visible or null", "reviewerBadge": "Elite, Top Contributor, Local Guide level, etc. or null", "photoCount": number of photos attached to review or null, "checkInCount": number of check-ins if visible or null, "platform": "google" or "yelp" or "tripadvisor" or "unknown", "suggestions": ["short description of each improvement suggestion found in the review"] or empty array}\n\nFor the text field: extract ONLY the customer review text. Do not include star ratings, dates, platform buttons, or any other UI elements.\nFor sentiment: determine if the review is positive, negative, or mixed.\nFor reviewerName: include the reviewer first name if visible in the screenshot, otherwise null.\nFor reviewerBadge: look for badges like Yelp Elite, Google Local Guide, TripAdvisor Top Contributor, etc.\nFor photoCount: count how many photos the reviewer attached to THIS review.\nFor checkInCount: if the platform shows check-ins for this reviewer at this business, include the count.\nFor platform: identify which review platform this screenshot is from.\nFor suggestions: identify any improvement suggestions, constructive feedback, or things the reviewer wished were different. Examples: "add background music", "more parking needed", "menu could have more options". Return empty array if no suggestions found.\n\nRespond with ONLY the JSON, no other text.',
            },
            {
              type: 'image_url',
              image_url: {
                url: 'data:' + mimeType + ';base64,' + base64Image,
              },
            },
          ],
        },
      ],
      max_tokens: 1000,
    });

    fs.unlinkSync(imagePath);

    var rawResponse = completion.choices[0].message.content;

    try {
      var cleaned = rawResponse.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
      var parsed = JSON.parse(cleaned);
      res.json({
        text: parsed.text,
        sentiment: parsed.sentiment || null,
        reviewerName: parsed.reviewerName || null,
        reviewerBadge: parsed.reviewerBadge || null,
        photoCount: parsed.photoCount || null,
        checkInCount: parsed.checkInCount || null,
        platform: parsed.platform || null,
        suggestions: parsed.suggestions || [],
      });
    } catch (parseError) {
      res.json({ text: rawResponse, sentiment: null, reviewerName: null, reviewerBadge: null, photoCount: null, checkInCount: null, platform: null, suggestions: [] });
    }
  } catch (error) {
    console.error('OCR Error:', error.message);
    res.status(500).json({ error: 'Failed to extract text from image' });
  }
});

// Helper: Build consistent signature block
function buildSignature(yourName, yourTitle, businessName) {
  if (!yourName) return '';
  var sig = yourName;
  if (yourTitle) sig += ', ' + yourTitle;
  if (businessName) sig += '\n' + businessName;
  return sig;
}

// Helper: Fix contact info and signature in response
function ensureContactAndSignature(response, isNegative, hasContact, contactMethod, contactInfo, yourName, yourTitle, businessName) {
  var text = response.trimEnd();

  if (isNegative && hasContact && text.indexOf(contactInfo) === -1) {
    var contactLine = contactMethod === 'email'
      ? 'Please reach out to me directly at ' + contactInfo + ' so we can resolve this.'
      : 'Please give me a call at ' + contactInfo + ' so we can resolve this.';
    var lines = text.split('\n');
    var lastLine = lines[lines.length - 1].trim();
    if (yourName && lastLine.indexOf(yourName) !== -1) {
      lines.splice(lines.length - 1, 0, '\n' + contactLine);
      text = lines.join('\n');
    } else {
      text = text + '\n\n' + contactLine;
    }
  }

  if (yourName) {
    var correctSig = buildSignature(yourName, yourTitle, businessName);
    var lines = text.split('\n');

    var sigStart = -1;
    for (var i = Math.max(0, lines.length - 4); i < lines.length; i++) {
      var trimmed = lines[i].trim();
      if (trimmed.indexOf(yourName) !== -1 ||
          trimmed.indexOf('Sincerely') !== -1 ||
          trimmed.indexOf('Best Regards') !== -1 ||
          trimmed.indexOf('Warm Regards') !== -1 ||
          trimmed.indexOf('Kind Regards') !== -1 ||
          trimmed.indexOf('Warmly') !== -1 ||
          trimmed.indexOf('Cheers') !== -1 ||
          trimmed.indexOf('All the best') !== -1 ||
          trimmed.indexOf('Respectfully') !== -1) {
        if (i > 0 && lines[i - 1].trim() === '') {
          sigStart = i - 1;
        } else {
          sigStart = i;
        }
        break;
      }
      if (trimmed.charAt(0) === '-' && trimmed.indexOf(yourName) !== -1) {
        if (i > 0 && lines[i - 1].trim() === '') {
          sigStart = i - 1;
        } else {
          sigStart = i;
        }
        break;
      }
    }

    if (sigStart !== -1) {
      var bodyText = lines.slice(0, sigStart).join('\n').trimEnd();
      if (bodyText.length > 0) {
        text = bodyText;
      }
    }

    text = text + '\n\n' + correctSig;
  }

  return text;
}

// Helper: Build reviewer context string for the prompt
function buildReviewerContext(reviewerName, reviewerBadge, photoCount, checkInCount, platform) {
  var parts = [];
  if (reviewerName) parts.push('The reviewer\'s name is ' + reviewerName + '.');
  if (reviewerBadge) parts.push('They have a "' + reviewerBadge + '" badge, which means they are an active and respected reviewer on the platform.');
  if (photoCount && photoCount > 0) parts.push('They included ' + photoCount + ' photo' + (photoCount > 1 ? 's' : '') + ' with their review.');
  if (checkInCount && checkInCount > 0) parts.push('They have checked in ' + checkInCount + ' time' + (checkInCount > 1 ? 's' : '') + '.');
  if (platform) parts.push('This review is on ' + platform + '.');
  return parts.join(' ');
}

// Helper: Platform-specific response style guidelines
function buildPlatformStyleGuidelines(platform) {
  if (!platform) return '';
  var lower = platform.toLowerCase();
  if (lower.indexOf('google') !== -1) {
    return '\nPlatform-Specific Guidelines (Google):\n' +
      '- Keep response concise (Google responses are most visible in search results)\n' +
      '- Include the business name naturally (helps with local SEO)\n' +
      '- Stay professional — Google reviews are highly visible to potential customers\n';
  }
  if (lower.indexOf('yelp') !== -1) {
    return '\nPlatform-Specific Guidelines (Yelp):\n' +
      '- Write a slightly longer, more detailed response (Yelp users expect engagement)\n' +
      '- Acknowledge specific details from the review — Yelp reviewers put effort into their posts\n' +
      '- Tone can be slightly more casual and conversational\n';
  }
  if (lower.indexOf('tripadvisor') !== -1 || lower.indexOf('trip advisor') !== -1) {
    return '\nPlatform-Specific Guidelines (TripAdvisor):\n' +
      '- Be warm and welcoming — TripAdvisor readers are often travelers planning visits\n' +
      '- Reference the experience rather than just the food\n' +
      '- Include a warm invitation to return\n';
  }
  if (lower.indexOf('facebook') !== -1) {
    return '\nPlatform-Specific Guidelines (Facebook):\n' +
      '- Keep it friendly and conversational — Facebook is a social platform\n' +
      '- Response can be slightly more informal than other platforms\n' +
      '- Short and warm works best\n';
  }
  return '';
}

// Helper: Detect subtle suggestions in positive reviews
function detectSuggestions(reviewText) {
  var suggestionIndicators = [
    'could use', 'would be nice', 'only complaint', 'only issue',
    'wish they', 'wish there', 'if only', 'my only', 'one thing',
    'suggestion', 'improve', 'could be better', 'room for',
    'would love to see', 'hope they', 'hopefully', 'missing',
    'lacking', 'needs more', 'not enough', 'but the', 'however',
    'although', 'though the', 'downside', 'except for',
    'would have been better', 'could have been', 'fell short'
  ];
  var lower = reviewText.toLowerCase();
  for (var i = 0; i < suggestionIndicators.length; i++) {
    if (lower.indexOf(suggestionIndicators[i]) !== -1) {
      return true;
    }
  }
  return false;
}

// Map tone names to prompt personality descriptions
function getToneInstructions(tone, isPositive) {
  var toneMap = {
    'Professional & Empathetic': {
      positive: 'Write in a professional but warm tone. Be gracious and appreciative without being over-the-top.',
      negative: 'Write in a professional, empathetic tone. Show you understand their frustration.'
    },
    'Warm & Friendly': {
      positive: 'Write like a genuinely excited owner who loves connecting with happy customers. Be warm, personal, and enthusiastic.',
      negative: 'Write in a warm, caring tone that shows real concern for their experience.'
    },
    'Brief & Direct': {
      positive: 'Keep it short and punchy -- 2-3 sentences max. Genuine but concise.',
      negative: 'Be brief and solution-oriented. Acknowledge, apologize, offer a fix. No fluff.'
    },
    'Recovery-Focused': {
      positive: 'Focus on building a lasting relationship. Invite them back, mention upcoming things to look forward to.',
      negative: 'Focus entirely on making this right. Be specific about next steps and how you will fix this.'
    },
    'Factual Correction': {
      positive: 'Be appreciative and use their review as a chance to share a fun detail or fact about what they enjoyed.',
      negative: 'Politely correct any inaccuracies while remaining respectful and offering to help.'
    },
    'Friendly & Energetic': {
      positive: 'Write like a fired-up local owner who genuinely loves what they do. High energy, exclamation points are okay, feel like a real person chatting back. Think neighborhood vibe.',
      negative: 'Be real and direct but with an upbeat energy about fixing the situation. Show passion for getting it right.'
    },
    'Community Builder': {
      positive: 'Write like someone building a local community around their spot. Make the reviewer feel like part of the family. Reference the neighborhood, regulars, the local scene.',
      negative: 'Emphasize that you are a community-focused business and this experience does not reflect your values. Invite them to give you another chance.'
    }
  };

  var key = isPositive ? 'positive' : 'negative';
  if (toneMap[tone]) return toneMap[tone][key];
  return toneMap['Professional & Empathetic'][key];
}

// Generate Endpoint - returns 5 variants for positive, 3 for negative
app.post('/generate', async function(req, res) {
  try {
    var body = req.body;
    var deviceId = body.deviceId;
    var isPro = body.isPro || false;

    // Input validation
    var reviewText = body.reviewText;
    if (!reviewText || typeof reviewText !== 'string' || reviewText.trim().length === 0) {
      return res.status(400).json({ error: 'reviewText is required' });
    }
    if (reviewText.length > 5000) {
      return res.status(400).json({ error: 'reviewText exceeds maximum length of 5000 characters' });
    }

    // Server-side generation limit enforcement
    if (deviceId && !isPro) {
      var currentCount = await getGenerationCount(deviceId);
      if (currentCount >= FREE_GENERATION_LIMIT) {
        return res.status(403).json({
          error: 'Generation limit reached',
          count: currentCount,
          limit: FREE_GENERATION_LIMIT,
        });
      }
    }

    var businessType = body.businessType;
    var reviewType = body.reviewType;
    var tone = body.tone;
    var businessName = body.businessName;
    var yourName = body.yourName;
    var yourTitle = body.yourTitle;
    var contactMethod = body.contactMethod;
    var contactInfo = body.contactInfo;
    var reviewerName = body.reviewerName || null;
    var reviewerBadge = body.reviewerBadge || null;
    var photoCount = body.photoCount || null;
    var checkInCount = body.checkInCount || null;
    var platform = body.platform || null;

    var isNegative = reviewType.toLowerCase().indexOf('negative') !== -1;
    var isPositive = reviewType.toLowerCase().indexOf('positive') !== -1;
    var isMixed = reviewType.toLowerCase().indexOf('mixed') !== -1;
    var hasContact = contactInfo && contactMethod;
    var hasSuggestion = detectSuggestions(reviewText);

    var signatureFormat = buildSignature(yourName, yourTitle, businessName);
    var reviewerContext = buildReviewerContext(reviewerName, reviewerBadge, photoCount, checkInCount, platform);
    var toneInstructions = getToneInstructions(tone, isPositive || isMixed);

    // Build base prompt
    var basePrompt = 'You are an AI Review Response Agent for a ' + businessType + '. You generate professional public responses to online customer reviews (Google, Yelp, Facebook, TripAdvisor, etc.).\n';
    basePrompt += 'Your responses should help maintain the business\'s reputation, show appreciation to customers, and demonstrate professionalism.\n\n';

    basePrompt += 'Write a response to this ' + reviewType.toLowerCase() + ' review.\n\n';

    if (businessName) basePrompt += 'Your restaurant/business is called ' + businessName + '.\n';
    if (yourName) basePrompt += 'Your name is ' + yourName + '.\n';
    if (yourTitle) basePrompt += 'Your title is ' + yourTitle + '.\n';

    if (reviewerContext) {
      basePrompt += '\nReviewer details: ' + reviewerContext + '\n';
    }

    basePrompt += '\nThe customer wrote:\n"' + reviewText + '"\n\n';

    // Tone guidelines
    basePrompt += 'Tone Guidelines:\n';
    basePrompt += '- ' + toneInstructions + '\n';
    basePrompt += '- Always be: warm, friendly, professional, calm, appreciative\n';
    basePrompt += '- Never sound: defensive, argumentative, robotic, corporate/legalistic, or condescending\n';
    basePrompt += '- Write like a thoughtful manager responding personally\n\n';

    // Style rules
    basePrompt += 'Style Rules:\n';
    basePrompt += '- Do NOT repeat menu items, products, or specific dishes mentioned in the review\n';
    basePrompt += '- Keep responses concise -- target 35-80 words, never exceed 120 words (excluding sign-off)\n';
    basePrompt += '- Do NOT start with "Dear Customer" -- address them naturally or skip a greeting\n';
    basePrompt += '- Do NOT use generic filler phrases like "We strive for excellence", "Your satisfaction is our top priority", "We value all our customers", or "Your feedback is important to us"\n';
    basePrompt += '- Write naturally, like a human\n';
    basePrompt += '- Do not apologize excessively\n';
    basePrompt += '- Do not admit legal fault or liability\n';
    basePrompt += '- Never blame the guest\n';
    basePrompt += '- Never mention photos, images, or visual content unless the review text itself explicitly references photos or images. Do not infer or assume photos exist based on reviewer badges or platform metadata.\n';
    basePrompt += '- Response must be ready to copy-paste as a public reply -- return ONLY the response text, no analysis, explanation, or headings\n\n';

    // Platform-specific adjustments
    var platformGuidelines = buildPlatformStyleGuidelines(platform);
    if (platformGuidelines) basePrompt += platformGuidelines + '\n';

    // Rating-based strategy
    if (isPositive) {
      basePrompt += 'Response Strategy (positive review):\n';
      basePrompt += '- Goal: Reinforce goodwill\n';
      basePrompt += '- Include: gratitude for the visit, appreciation for the kind words, a welcoming closing line\n';
      basePrompt += '- Structure: Thank → Appreciate → Welcome back\n';

      if (reviewerName) {
        basePrompt += '- Address the reviewer by their first name naturally\n';
      }
      if (reviewerBadge) {
        basePrompt += '- Acknowledge their reviewer status naturally (e.g., "Thanks for the detailed review" or "We appreciate active reviewers like you sharing their experience") -- do NOT just say "thanks for being Elite" awkwardly\n';
      }
      if (photoCount && photoCount > 0) {
        basePrompt += '- Thank them for sharing photos -- mention how photos help other customers discover you (e.g., "Love the photos you shared -- they really capture the vibe!")\n';
      }
      if (checkInCount && checkInCount > 1) {
        basePrompt += '- Acknowledge that they are a repeat visitor and how much that means to you\n';
      }
      if (hasSuggestion) {
        basePrompt += '- IMPORTANT: The reviewer included a suggestion or constructive feedback within their positive review. Acknowledge it specifically and respond to it genuinely.\n';
      }
    }

    if (isMixed) {
      basePrompt += 'Response Strategy (mixed review):\n';
      basePrompt += '- Goal: Show attentiveness and openness to improvement\n';
      basePrompt += '- Include: thank them for the feedback, recognize mixed experience, encourage another visit\n';
      basePrompt += '- Structure: Thank → Acknowledge → Improve → Invite back\n';

      if (reviewerName) {
        basePrompt += '- Address the reviewer by their first name naturally\n';
      }
      if (reviewerBadge) {
        basePrompt += '- Acknowledge their reviewer status naturally\n';
      }
      if (photoCount && photoCount > 0) {
        basePrompt += '- Thank them for sharing photos\n';
      }
      if (checkInCount && checkInCount > 1) {
        basePrompt += '- Acknowledge that they are a repeat visitor\n';
      }
      if (hasSuggestion) {
        basePrompt += '- IMPORTANT: The reviewer included a suggestion or constructive feedback. Acknowledge it specifically and respond to it genuinely.\n';
      }
    }

    if (isNegative) {
      basePrompt += 'Response Strategy (negative review):\n';
      basePrompt += '- Goal: Protect reputation and move conversation offline\n';
      basePrompt += '- Include: calm acknowledgement, professional tone, invitation to contact management privately\n';
      basePrompt += '- Structure: Acknowledge → Express concern → Invite contact\n';
      basePrompt += '- Never: argue, accuse, or debate facts publicly\n';
      if (reviewerName) {
        basePrompt += '- Address the reviewer by their first name\n';
      }
    }

    // Edge cases
    basePrompt += '\nEdge Cases:\n';
    basePrompt += '- If the review has no comment text, respond with a brief thank-you\n';
    basePrompt += '- If the review contains extremely hostile language, remain calm and professional\n';
    basePrompt += '- If the review contains false accusations, do not argue publicly -- offer to discuss offline\n';
    basePrompt += '- If the review is about service complaints, show concern and willingness to improve\n';

    if (isNegative && hasContact) {
      basePrompt += '\n- CRITICAL: The response MUST include "' + contactInfo + '" as the contact ' + contactMethod + '. Do NOT invent a different email/phone. Use ONLY "' + contactInfo + '".\n';
    }

    if (yourName) {
      basePrompt += '\n- Sign off with EXACTLY this on the last lines (no "Best Regards", "Sincerely", "Warm Regards", etc.):\n' + signatureFormat + '\n';
    } else {
      basePrompt += '\n- Do not include a sign-off\n';
    }

    basePrompt += '\nCore Principle: The response should always make readers think: "This business is professional, attentive, and cares about its guests."\n';

    // Build variant prompts
    var variantPrompts = [];

    if (isPositive || isMixed) {
      // 5 strategically different positive variants

      // V1: Short & Sweet (for quick posting)
      var v1 = basePrompt;
      v1 += '- LENGTH: Keep this VERY short -- 2-3 sentences max. Quick, punchy, perfect for fast posting.\n';
      v1 += '\nWrite a short and sweet thank-you response:';

      // V2: Detailed (echoes every point)
      var v2 = basePrompt;
      v2 += '- LENGTH: This can be longer, 5-7 sentences. Touch on EVERY positive point the reviewer mentioned. Make them feel truly heard.\n';
      v2 += '\nWrite a detailed response that addresses each thing the customer praised:';

      // V3: Proactive about suggestions OR behind-the-scenes
      var v3 = basePrompt;
      if (hasSuggestion) {
        v3 += '- LENGTH: 4-5 sentences.\n';
        v3 += '\nWrite a response that is PROACTIVE about the suggestion. Instead of saying you will "consider" it, show enthusiasm and hint you are already working on it or exploring it. Make the reviewer feel like their feedback drives real change.\nWrite this response now:';
      } else {
        v3 += '- LENGTH: 4-5 sentences.\n';
        v3 += '\nWrite a response with a personal, conversational angle. Share a brief behind-the-scenes detail or fun fact about something they mentioned.\nWrite this response now:';
      }

      // V4: Gentle upsell/invite back
      var v4 = basePrompt;
      v4 += '- LENGTH: 3-5 sentences.\n';
      v4 += '\nWrite a response that ends with a gentle upsell or invitation. Suggest a specific item they should try next time (make one up if needed based on the business type), mention a seasonal special, or tease an upcoming addition. Make them excited to come back for something specific.\nWrite this response now:';

      // V5: Celebrates growth/community OR reviewer engagement
      var v5 = basePrompt;
      if (reviewerBadge || (photoCount && photoCount > 0) || (checkInCount && checkInCount > 1)) {
        v5 += '- LENGTH: 4-5 sentences.\n';
        var engagementAspects = [];
        if (photoCount && photoCount > 0) engagementAspects.push('their photos');
        if (checkInCount && checkInCount > 1) engagementAspects.push('their repeat visits');
        if (reviewerBadge) engagementAspects.push('their active presence on the platform');
        v5 += '\nWrite a response that especially celebrates this reviewer\'s engagement -- ' + engagementAspects.join(', ') + '. Make them feel valued as a community member and part of the family, not just a customer.\nWrite this response now:';
      } else {
        v5 += '- LENGTH: 4-5 sentences.\n';
        v5 += '\nWrite a response that highlights your business growth, your pride in what you have built, or how the community has supported you. If the reviewer mentioned expansion, growth, or that you are new, lean into that excitement. Make the reviewer feel like part of your journey.\nWrite this response now:';
      }

      variantPrompts = [v1, v2, v3, v4, v5];
    } else {
      // 3 negative review variants (unchanged)
      variantPrompts = [
        basePrompt + '\nWrite a response now:',
        basePrompt + '\nWrite a DIFFERENT response with a slightly different angle or opening:',
        basePrompt + '\nWrite ANOTHER unique response, varying the structure and wording from typical responses:',
      ];
    }

    var results = await Promise.allSettled(
      variantPrompts.map(function(prompt) {
        return openai.chat.completions.create({
          model: 'gpt-4o-mini',
          messages: [{ role: 'user', content: prompt }],
          max_tokens: 500,
          temperature: 0.85,
        });
      })
    );

    var responses = [];
    var totalInputTokens = 0;
    var totalOutputTokens = 0;
    for (var i = 0; i < results.length; i++) {
      if (results[i].status === 'fulfilled') {
        var completion = results[i].value;
        var text = completion.choices[0].message.content;
        text = ensureContactAndSignature(text, isNegative, hasContact, contactMethod, contactInfo, yourName, yourTitle, businessName);
        responses.push(text);
        totalInputTokens += completion.usage ? completion.usage.prompt_tokens : 0;
        totalOutputTokens += completion.usage ? completion.usage.completion_tokens : 0;
      } else {
        console.error('[generate] Variant ' + i + ' failed:', results[i].reason?.message);
      }
    }
    var generationCostMicroUsd = calcCostMicroUsd(totalInputTokens, totalOutputTokens);

    if (responses.length === 0) {
      return res.status(500).json({ error: 'All response variants failed to generate' });
    }

    // Generate quick reply for positive reviews
    var quickReply = null;
    if ((isPositive || isMixed) && yourName) {
      var firstName = reviewerName ? reviewerName.split(' ')[0].replace(/[^a-zA-Z]/g, '') : null;
      var highlights = [];
      var lower = reviewText.toLowerCase();
      if (lower.indexOf('food') !== -1 || lower.indexOf('delicious') !== -1 || lower.indexOf('tasty') !== -1) highlights.push('the food');
      if (lower.indexOf('service') !== -1 || lower.indexOf('staff') !== -1 || lower.indexOf('kind') !== -1 || lower.indexOf('friendly') !== -1) highlights.push('the team');
      if (lower.indexOf('clean') !== -1 || lower.indexOf('ambien') !== -1 || lower.indexOf('atmosphere') !== -1 || lower.indexOf('vibe') !== -1) highlights.push('the vibe');
      if (lower.indexOf('presentation') !== -1 || lower.indexOf('beautiful') !== -1) highlights.push('the presentation');
      if (lower.indexOf('expand') !== -1 || lower.indexOf('new') !== -1 || lower.indexOf('location') !== -1) highlights.push('the new spot');
      if (lower.indexOf('customiz') !== -1 || lower.indexOf('individual') !== -1 || lower.indexOf('build') !== -1) highlights.push('the customization');

      var highlightText = highlights.length > 0 ? highlights.slice(0, 2).join(' & ') : 'your visit';

      if (firstName) {
        quickReply = 'Thanks ' + firstName + '! Glad you loved ' + highlightText + ' -- see you soon! -' + yourName;
      } else {
        quickReply = 'Thanks so much! Glad you loved ' + highlightText + ' -- see you soon! -' + yourName;
      }
    }

    // Increment server-side generation counter and log cost
    var newCount = 0;
    if (deviceId) {
      newCount = await incrementGeneration(deviceId, generationCostMicroUsd);
    }

    res.json({
      responses: responses,
      quickReply: quickReply,
      variantLabels: (isPositive || isMixed)
        ? ['Short & Sweet', 'Detailed', hasSuggestion ? 'Proactive Fix' : 'Personal', 'Invite Back', (reviewerBadge || photoCount) ? 'Fan Love' : 'Our Story']
        : ['Option 1', 'Option 2', 'Option 3'],
      generationCount: newCount,
      generationsRemaining: isPro ? 999999 : Math.max(0, FREE_GENERATION_LIMIT - newCount),
    });
  } catch (error) {
    console.error('Error:', error.message);
    res.status(500).json({ error: 'Failed to generate response' });
  }
});

// Admin stats endpoint — secured by ADMIN_TOKEN env var
app.get('/admin/stats', async function(req, res) {
  var token = req.headers['x-admin-token'];
  if (!token || token !== process.env.ADMIN_TOKEN) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  try {
    var month = req.query.month || getCurrentMonth();
    var stats = await pool.query(`
      SELECT
        COUNT(*) AS active_devices,
        SUM(count) AS total_generations,
        SUM(cost_micro_usd) AS total_cost_micro_usd,
        AVG(count) AS avg_generations_per_device,
        MAX(count) AS max_generations_device
      FROM generations
      WHERE month = $1
    `, [month]);
    var row = stats.rows[0];
    var totalCostUsd = (parseInt(row.total_cost_micro_usd) || 0) / 1000000;
    res.json({
      month: month,
      activeDevices: parseInt(row.active_devices) || 0,
      totalGenerations: parseInt(row.total_generations) || 0,
      avgGenerationsPerDevice: parseFloat(row.avg_generations_per_device) || 0,
      maxGenerationsDevice: parseInt(row.max_generations_device) || 0,
      totalCostUsd: totalCostUsd.toFixed(4),
      estimatedCostPerGeneration: row.total_generations > 0
        ? (totalCostUsd / parseInt(row.total_generations)).toFixed(6)
        : '0.000000',
    });
  } catch (err) {
    console.error('[admin/stats] Error:', err.message);
    res.status(500).json({ error: 'Failed to fetch stats' });
  }
});

// ── Centralized error handler ────────────────────────────────
app.use(function(err, req, res, next) {
  if (err.message === 'Only image files are allowed' || err.code === 'LIMIT_FILE_SIZE') {
    return res.status(400).json({ error: err.message || 'File too large (max 10MB)' });
  }
  console.error('[unhandled]', err.message);
  res.status(500).json({ error: 'Internal server error' });
});

var PORT = process.env.PORT || 3000;
app.listen(PORT, '0.0.0.0', () => {
  // Validate required env vars
  var required = ['OPENAI_API_KEY', 'DATABASE_URL'];
  var missing = required.filter(function(v) { return !process.env[v]; });
  if (missing.length > 0) {
    console.error('[STARTUP] Missing required env vars:', missing.join(', '));
  }
  console.log(`Server running on http://0.0.0.0:${PORT}`);
});