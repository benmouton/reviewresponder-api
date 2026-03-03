var express = require('express');
var cors = require('cors');
var OpenAI = require('openai');
var multer = require('multer');
var fs = require('fs');
var path = require('path');

var app = express();
app.use(cors());
app.use(express.json());

var upload = multer({ dest: 'uploads/' });

var openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// OCR Endpoint - extracts reviewer metadata and detects suggestions
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

    if (sigStart !== -1 && sigStart >= 2 && lines.slice(0, sigStart).join('').trim().length > 0) {
      text = lines.slice(0, sigStart).join('\n').trimEnd();
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
    var businessType = body.businessType;
    var reviewType = body.reviewType;
    var tone = body.tone;
    var businessName = body.businessName;
    var yourName = body.yourName;
    var yourTitle = body.yourTitle;
    var contactMethod = body.contactMethod;
    var contactInfo = body.contactInfo;
    var reviewText = body.reviewText;
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
    var basePrompt = 'You are a ' + businessType + ' owner who personally responds to online reviews. Write a response to this ' + reviewType.toLowerCase() + '.\n\n';

    if (businessName) basePrompt += 'Your restaurant/business is called ' + businessName + '.\n';
    if (yourName) basePrompt += 'Your name is ' + yourName + '.\n';
    if (yourTitle) basePrompt += 'Your title is ' + yourTitle + '.\n';

    if (reviewerContext) {
      basePrompt += '\nReviewer details: ' + reviewerContext + '\n';
    }

    basePrompt += '\nThe customer wrote:\n"' + reviewText + '"\n\n';

    basePrompt += 'Guidelines:\n';
    basePrompt += '- Tone instructions: ' + toneInstructions + '\n';
    basePrompt += '- Sound like a real person, not a corporate template\n';
    basePrompt += '- Reference specific things the customer mentioned\n';
    basePrompt += '- Do NOT start with "Dear Customer" -- address them naturally or skip a greeting\n';
    basePrompt += '- Do NOT use phrases like "We value all our customers" or "Your feedback is important to us"\n';

    if (isPositive || isMixed) {
      basePrompt += '- Be warm and genuine, mention what they enjoyed specifically\n';

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

    if (isNegative) {
      basePrompt += '- Keep it concise (4-7 sentences)\n';
      basePrompt += '- Acknowledge the issue specifically, take ownership, offer to make it right\n';
      if (reviewerName) {
        basePrompt += '- Address the reviewer by their first name\n';
      }
    }

    if (isNegative && hasContact) {
      basePrompt += '- CRITICAL: The response MUST include "' + contactInfo + '" as the contact ' + contactMethod + '. Do NOT invent a different email/phone. Use ONLY "' + contactInfo + '".\n';
    }

    if (yourName) {
      basePrompt += '- Sign off with EXACTLY this on the last lines (no "Best Regards", "Sincerely", "Warm Regards", etc.):\n' + signatureFormat + '\n';
    } else {
      basePrompt += '- Do not include a sign-off\n';
    }

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
        v5 += '\nWrite a response that especially celebrates this reviewer\'s engagement -- their photos, their repeat visits, or their active presence on the platform. Make them feel valued as a community member and part of the family, not just a customer.\nWrite this response now:';
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

    var results = await Promise.all(
      variantPrompts.map(function(prompt) {
        return openai.chat.completions.create({
          model: 'gpt-4',
          messages: [{ role: 'user', content: prompt }],
          max_tokens: 500,
          temperature: 0.85,
        });
      })
    );

    var responses = results.map(function(result) {
      var text = result.choices[0].message.content;
      text = ensureContactAndSignature(text, isNegative, hasContact, contactMethod, contactInfo, yourName, yourTitle, businessName);
      return text;
    });

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

    res.json({
      responses: responses,
      quickReply: quickReply,
      variantLabels: (isPositive || isMixed)
        ? ['Short & Sweet', 'Detailed', hasSuggestion ? 'Proactive Fix' : 'Personal', 'Invite Back', (reviewerBadge || photoCount) ? 'Fan Love' : 'Our Story']
        : ['Option 1', 'Option 2', 'Option 3'],
    });
  } catch (error) {
    console.error('Error:', error.message);
    res.status(500).json({ error: 'Failed to generate response' });
  }
});

var PORT = 3000;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
});
