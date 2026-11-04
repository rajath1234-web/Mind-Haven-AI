"""
MindHaven AI – Mental Health Awareness & Support Companion
==========================================================
A Flask-based AI mental health support application using Groq API.
Designed for AI/ML hackathons – beginner-friendly and well-commented.

Run:  python app.py
URL:  http://127.0.0.1:5000
"""

import os
import json
import re
from datetime import datetime
from flask import Flask, request, jsonify, session, render_template_string
from groq import Groq
from dotenv import load_dotenv

# ── Load environment variables from .env file ──────────────────────────────────
load_dotenv()

app = Flask(__name__)
# Secret key for Flask session (change this in production!)
app.secret_key = os.getenv("SECRET_KEY", "mindhaven-secret-2024")

# ── Groq API configuration ─────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")   # free open-source model

# Initialise Groq client (only if key is present)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ── High-risk / crisis keyword filter ─────────────────────────────────────────
# These keywords trigger the emergency-referral protocol immediately.
HIGH_RISK_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life", "want to die",
    "no reason to live", "self-harm", "self harm", "cutting myself",
    "hurt myself", "overdose", "can't go on", "cannot go on",
    "better off dead", "not worth living", "end it all",
]

# Moderate-distress keywords (raise stress level but don't trigger crisis mode)
MODERATE_RISK_KEYWORDS = [
    "hopeless", "worthless", "depressed", "depression", "anxious", "anxiety",
    "panic attack", "panic", "overwhelmed", "exhausted", "burnout",
    "can't cope", "cannot cope", "breaking down", "falling apart",
    "no one cares", "nobody cares", "alone", "lonely", "isolated",
]

# ── Mock helpline data (real verified numbers) ─────────────────────────────────
HELPLINES = [
    {"name": "iCall (India)",           "number": "9152987821",  "available": "Mon–Sat, 8am–10pm"},
    {"name": "Vandrevala Foundation",   "number": "1860-2662-345","available": "24/7"},
    {"name": "NIMHANS Helpline",        "number": "080-46110007", "available": "24/7"},
    {"name": "Snehi (India)",           "number": "044-24640050", "available": "Daily, 8am–10pm"},
    {"name": "Crisis Text Line (US)",   "number": "Text HOME to 741741", "available": "24/7"},
    {"name": "Samaritans (UK)",         "number": "116 123",     "available": "24/7"},
    {"name": "Lifeline (Australia)",    "number": "13 11 14",    "available": "24/7"},
    {"name": "International Assoc. for Suicide Prevention",
                                        "number": "https://www.iasp.info/resources/Crisis_Centres/",
                                        "available": "Global directory"},
]

# ── Coping resources ───────────────────────────────────────────────────────────
COPING_RESOURCES = {
    "breathing": {
        "title": "4-7-8 Breathing Exercise",
        "description": "Inhale for 4 counts, hold for 7, exhale for 8. Repeat 4 times.",
        "duration": "~5 minutes",
        "category": "Breathing",
    },
    "grounding": {
        "title": "5-4-3-2-1 Grounding Technique",
        "description": "Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
        "duration": "~3 minutes",
        "category": "Grounding",
    },
    "journaling": {
        "title": "Gratitude Journaling",
        "description": "Write 3 things you're grateful for today, no matter how small.",
        "duration": "~10 minutes",
        "category": "Journaling",
    },
    "mindfulness": {
        "title": "Body Scan Meditation",
        "description": "Slowly shift attention from feet to head, noticing sensations without judgment.",
        "duration": "~15 minutes",
        "category": "Mindfulness",
    },
    "movement": {
        "title": "Gentle Movement Break",
        "description": "5 minutes of light stretching or a short walk outside to reset your nervous system.",
        "duration": "~5 minutes",
        "category": "Movement",
    },
    "self_compassion": {
        "title": "Self-Compassion Pause",
        "description": "Place a hand on your heart, say: 'This is a moment of suffering. I'm not alone. May I be kind to myself.'",
        "duration": "~2 minutes",
        "category": "Self-Compassion",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  SAFETY FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def assess_risk_level(text: str) -> str:
    """
    Keyword-based risk assessment.
    Returns: 'high', 'moderate', or 'low'
    """
    text_lower = text.lower()
    for kw in HIGH_RISK_KEYWORDS:
        if kw in text_lower:
            return "high"
    for kw in MODERATE_RISK_KEYWORDS:
        if kw in text_lower:
            return "moderate"
    return "low"


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE LLM HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def generate_response(messages: list, temperature: float = 0.7) -> str:
    """
    Send a list of messages to the Groq API and return the assistant reply.
    Falls back to a safe static message if the API key is missing or an error occurs.
    """
    if not groq_client:
        return (
            "⚠️ Groq API key is not configured. Please set GROQ_API_KEY in your .env file. "
            "I'm still here for you — please use the resources on this page if you need support."
        )
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=512,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        # Never surface raw API errors to the user — keep it safe and calm
        print(f"[Groq API Error] {e}")
        return (
            "I'm having a little trouble connecting right now, but I'm still here with you. "
            "Please take a gentle breath. If you need immediate support, the helplines below are available 24/7."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT 1 – Empathetic Listener & Detection Agent
# ═══════════════════════════════════════════════════════════════════════════════

LISTENER_SYSTEM_PROMPT = """You are MindHaven, a compassionate and non-judgmental AI mental health support companion.
Your role is to listen deeply, validate feelings, and provide emotional support.

Guidelines:
- Use warm, empathetic, and gentle language at all times.
- Practice active listening: reflect back what the user says, ask open-ended follow-up questions.
- Never minimise, dismiss, or offer unsolicited advice unless asked.
- Never diagnose, prescribe medication, or make clinical judgements.
- Be culturally sensitive and inclusive.
- Keep responses concise (3–5 sentences) unless the user clearly wants more.
- Always end your response with the following disclaimer on a new line:
  "---\n⚠️ Disclaimer: I am an AI companion, not a licensed therapist. This is not a substitute for professional mental health care."
"""

def empathetic_listener_agent(user_message: str, chat_history: list) -> str:
    """
    Agent 1: Listens to the user, detects emotional tone, and responds with empathy.
    Accepts full chat history for context-aware responses.
    """
    # Build the message list: system prompt + history + new user message
    messages = [{"role": "system", "content": LISTENER_SYSTEM_PROMPT}]
    messages.extend(chat_history[-10:])   # keep last 10 turns for context
    messages.append({"role": "user", "content": user_message})

    return generate_response(messages, temperature=0.75)


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT 2 – Proactive Care & Resource Agent
# ═══════════════════════════════════════════════════════════════════════════════

CARE_SYSTEM_PROMPT = """You are MindHaven Care, a proactive wellness coach specialising in evidence-based
self-care strategies for mental health.

Your role:
- Provide personalised, practical coping strategies based on the user's current emotional state.
- Suggest mindfulness exercises, breathing techniques, movement, journaling prompts, or grounding exercises.
- Be encouraging, specific, and actionable — not generic.
- Keep suggestions brief (2–4 bullet points) and achievable right now.
- End with a brief motivational sentence that feels genuine, not clichéd.
"""

def proactive_care_agent(user_message: str, risk_level: str) -> str:
    """
    Agent 2: Generates personalised self-care recommendations based on risk level.
    """
    context = f"The user shared: '{user_message}'\nDetected stress level: {risk_level}.\n"
    if risk_level == "high":
        context += "Focus on immediate safety, grounding, and encouraging professional help."
    elif risk_level == "moderate":
        context += "Focus on calming techniques, self-compassion, and emotional regulation."
    else:
        context += "Focus on positive habits and preventive wellness practices."

    messages = [
        {"role": "system", "content": CARE_SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    return generate_response(messages, temperature=0.65)


# ═══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR – Routes input and coordinates both agents
# ═══════════════════════════════════════════════════════════════════════════════

def orchestrate(user_message: str, chat_history: list) -> dict:
    """
    Main orchestrator:
    1. Runs the safety filter to determine risk level.
    2. Calls Agent 1 (Listener) for empathetic response.
    3. Calls Agent 2 (Care) for self-care suggestions.
    4. If high-risk → triggers emergency-referral protocol.

    Returns a dict with keys: listener_reply, care_reply, risk_level, show_emergency
    """
    risk_level   = assess_risk_level(user_message)
    show_emergency = risk_level == "high"

    listener_reply = empathetic_listener_agent(user_message, chat_history)
    care_reply     = proactive_care_agent(user_message, risk_level)

    return {
        "listener_reply": listener_reply,
        "care_reply":     care_reply,
        "risk_level":     risk_level,
        "show_emergency": show_emergency,
        "helplines":      HELPLINES if show_emergency else [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML TEMPLATES (all inline via render_template_string)
# ═══════════════════════════════════════════════════════════════════════════════

BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>MindHaven AI – {% block title %}Mental Health Companion{% endblock %}</title>

  <!-- Bootstrap 5 CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <!-- Bootstrap Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet"/>

  <style>
    /* ── Global palette: soft pastels & calming tones ── */
    :root {
      --mh-bg:         #f0f4f8;
      --mh-sidebar:    #e8f0fe;
      --mh-primary:    #5b8dee;
      --mh-secondary:  #7ec8a4;
      --mh-accent:     #f9a8d4;
      --mh-danger:     #f87171;
      --mh-text:       #374151;
      --mh-muted:      #6b7280;
      --mh-card:       #ffffff;
      --mh-border:     #d1dae8;
      --mh-chat-user:  #dbeafe;
      --mh-chat-bot:   #f0fdf4;
    }

    body {
      background: var(--mh-bg);
      color: var(--mh-text);
      font-family: 'Segoe UI', system-ui, sans-serif;
      min-height: 100vh;
    }

    /* ── Sidebar ── */
    #sidebar {
      width: 260px;
      min-height: 100vh;
      background: var(--mh-sidebar);
      border-right: 1px solid var(--mh-border);
      position: fixed;
      top: 0; left: 0;
      display: flex;
      flex-direction: column;
      z-index: 100;
      transition: transform .3s ease;
    }

    #sidebar .brand {
      padding: 1.5rem 1.25rem 1rem;
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--mh-primary);
      border-bottom: 1px solid var(--mh-border);
    }

    #sidebar .brand span { color: var(--mh-secondary); }

    #sidebar .nav-link {
      color: var(--mh-text);
      border-radius: 8px;
      margin: 2px 10px;
      padding: .6rem .9rem;
      display: flex;
      align-items: center;
      gap: .55rem;
      transition: background .18s;
    }

    #sidebar .nav-link:hover,
    #sidebar .nav-link.active {
      background: var(--mh-primary);
      color: #fff;
    }

    #sidebar .emergency-btn {
      margin: auto 1rem 1.5rem;
      background: var(--mh-danger);
      color: #fff;
      border: none;
      border-radius: 10px;
      padding: .75rem;
      font-weight: 600;
      text-align: center;
      text-decoration: none;
      display: block;
      transition: opacity .2s;
    }

    #sidebar .emergency-btn:hover { opacity: .88; color: #fff; }

    /* ── Main content wrapper ── */
    #main-content {
      margin-left: 260px;
      padding: 2rem;
      min-height: 100vh;
    }

    /* ── Cards ── */
    .mh-card {
      background: var(--mh-card);
      border: 1px solid var(--mh-border);
      border-radius: 14px;
      padding: 1.5rem;
      box-shadow: 0 2px 10px rgba(91,141,238,.07);
    }

    /* ── Chat ── */
    #chat-box {
      height: 430px;
      overflow-y: auto;
      border: 1px solid var(--mh-border);
      border-radius: 12px;
      background: #fff;
      padding: 1rem;
      scroll-behavior: smooth;
    }

    .chat-bubble {
      max-width: 78%;
      padding: .7rem 1rem;
      border-radius: 14px;
      margin-bottom: .75rem;
      line-height: 1.55;
      font-size: .93rem;
    }

    .bubble-user {
      background: var(--mh-chat-user);
      margin-left: auto;
      border-bottom-right-radius: 4px;
    }

    .bubble-bot {
      background: var(--mh-chat-bot);
      margin-right: auto;
      border-bottom-left-radius: 4px;
    }

    .bubble-care {
      background: #fef9c3;
      margin-right: auto;
      border-bottom-left-radius: 4px;
      border-left: 3px solid #fbbf24;
    }

    .bubble-emergency {
      background: #fee2e2;
      border: 1px solid #fca5a5;
      border-radius: 12px;
      padding: 1rem;
      margin-bottom: .75rem;
    }

    /* ── Risk badge ── */
    .risk-badge {
      font-size: .72rem;
      padding: .2em .55em;
      border-radius: 999px;
      font-weight: 600;
    }

    .risk-low      { background: #d1fae5; color: #065f46; }
    .risk-moderate { background: #fef3c7; color: #92400e; }
    .risk-high     { background: #fee2e2; color: #991b1b; }

    /* ── Stat cards on dashboard ── */
    .stat-card {
      border-radius: 14px;
      padding: 1.3rem 1.5rem;
      color: #fff;
      min-height: 110px;
    }

    .stat-blue  { background: linear-gradient(135deg, #5b8dee, #93c5fd); }
    .stat-green { background: linear-gradient(135deg, #7ec8a4, #bbf7d0); color: #1a4731; }
    .stat-pink  { background: linear-gradient(135deg, #f9a8d4, #fbcfe8); color: #831843; }
    .stat-amber { background: linear-gradient(135deg, #fbbf24, #fde68a); color: #78350f; }

    /* ── Resource cards ── */
    .resource-card {
      border-left: 4px solid var(--mh-primary);
      border-radius: 10px;
      background: #fff;
      padding: 1rem 1.25rem;
      margin-bottom: 1rem;
      box-shadow: 0 1px 6px rgba(0,0,0,.06);
      transition: transform .18s;
    }

    .resource-card:hover { transform: translateY(-2px); }

    /* ── Responsive sidebar toggle ── */
    @media (max-width: 768px) {
      #sidebar { transform: translateX(-100%); }
      #sidebar.open { transform: translateX(0); }
      #main-content { margin-left: 0; padding: 1rem; }
      .sidebar-toggle { display: block !important; }
    }

    .sidebar-toggle { display: none; }

    /* ── Typing indicator ── */
    .typing-dots span {
      width: 7px; height: 7px;
      border-radius: 50%;
      background: var(--mh-primary);
      display: inline-block;
      animation: bounce .9s infinite;
    }

    .typing-dots span:nth-child(2) { animation-delay: .15s; }
    .typing-dots span:nth-child(3) { animation-delay: .30s; }

    @keyframes bounce {
      0%, 80%, 100% { transform: scale(0.6); opacity:.5; }
      40%            { transform: scale(1.0); opacity:1; }
    }

    /* ── Scrollbar ── */
    #chat-box::-webkit-scrollbar { width: 5px; }
    #chat-box::-webkit-scrollbar-thumb { background: var(--mh-border); border-radius: 99px; }
  </style>
</head>
<body>

<!-- ── Sidebar ───────────────────────────────────────────── -->
<nav id="sidebar">
  <div class="brand">
    <i class="bi bi-heart-pulse-fill me-1"></i>Mind<span>Haven</span> AI
  </div>

  <ul class="nav flex-column mt-3 flex-grow-1">
    <li class="nav-item">
      <a class="nav-link {{ 'active' if active_page=='home' }}" href="/">
        <i class="bi bi-house-door-fill"></i> Dashboard
      </a>
    </li>
    <li class="nav-item">
      <a class="nav-link {{ 'active' if active_page=='chat' }}" href="/chat">
        <i class="bi bi-chat-heart-fill"></i> Safe Space Chat
      </a>
    </li>
    <li class="nav-item">
      <a class="nav-link {{ 'active' if active_page=='resources' }}" href="/resources">
        <i class="bi bi-journal-bookmark-fill"></i> Coping Resources
      </a>
    </li>
    <li class="nav-item">
      <a class="nav-link {{ 'active' if active_page=='about' }}" href="/about">
        <i class="bi bi-info-circle-fill"></i> About
      </a>
    </li>
  </ul>

  <!-- Prominent emergency button always visible in sidebar -->
  <a href="/help" class="emergency-btn">
    <i class="bi bi-telephone-fill me-1"></i> Get Help Now
  </a>
</nav>

<!-- ── Mobile toggle ────────────────────────────────────── -->
<button class="sidebar-toggle btn btn-light m-2 position-fixed"
        style="top:0;left:0;z-index:200;" onclick="toggleSidebar()">
  <i class="bi bi-list fs-4"></i>
</button>

<!-- ── Main content ──────────────────────────────────────── -->
<main id="main-content">
  {% block content %}{% endblock %}
</main>

<!-- Bootstrap 5 JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
  function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
  }
</script>
{% block extra_js %}{% endblock %}
</body>
</html>"""


# ── Home / Dashboard template ─────────────────────────────────────────────────
HOME_HTML = BASE_HTML.replace("{% block title %}Mental Health Companion{% endblock %}", "Dashboard") \
    .replace("{% block content %}{% endblock %}", """
<div class="d-flex justify-content-between align-items-center mb-4">
  <div>
    <h2 class="fw-bold mb-0">Welcome to MindHaven AI 💙</h2>
    <p class="text-muted">Your safe space for mental wellness support.</p>
  </div>
  <span class="badge bg-success">System Online</span>
</div>

<!-- Stat cards -->
<div class="row g-3 mb-4">
  <div class="col-sm-6 col-xl-3">
    <div class="stat-card stat-blue">
      <div class="fs-4 fw-bold"><i class="bi bi-chat-heart me-2"></i>AI Agents</div>
      <div class="mt-1 small">2 specialised companions active</div>
    </div>
  </div>
  <div class="col-sm-6 col-xl-3">
    <div class="stat-card stat-green">
      <div class="fs-4 fw-bold"><i class="bi bi-shield-check me-2"></i>Safe</div>
      <div class="mt-1 small">Crisis keyword filter enabled</div>
    </div>
  </div>
  <div class="col-sm-6 col-xl-3">
    <div class="stat-card stat-pink">
      <div class="fs-4 fw-bold"><i class="bi bi-telephone me-2"></i>8 Helplines</div>
      <div class="mt-1 small">Available across India, US, UK, AU</div>
    </div>
  </div>
  <div class="col-sm-6 col-xl-3">
    <div class="stat-card stat-amber">
      <div class="fs-4 fw-bold"><i class="bi bi-journal-text me-2"></i>6 Exercises</div>
      <div class="mt-1 small">Evidence-based coping tools</div>
    </div>
  </div>
</div>

<!-- Quick actions -->
<div class="row g-3 mb-4">
  <div class="col-md-6">
    <div class="mh-card h-100">
      <h5 class="fw-semibold mb-2"><i class="bi bi-chat-heart-fill text-primary me-2"></i>Start a Conversation</h5>
      <p class="text-muted small">Talk to our empathetic AI listener. Share how you're feeling — no judgment, ever.</p>
      <a href="/chat" class="btn btn-primary rounded-pill px-4">Open Safe Space Chat</a>
    </div>
  </div>
  <div class="col-md-6">
    <div class="mh-card h-100">
      <h5 class="fw-semibold mb-2"><i class="bi bi-journal-bookmark-fill text-success me-2"></i>Explore Resources</h5>
      <p class="text-muted small">Browse mindfulness exercises, breathing techniques, and self-care routines.</p>
      <a href="/resources" class="btn btn-success rounded-pill px-4">View Resources</a>
    </div>
  </div>
</div>

<!-- About the agents -->
<div class="mh-card mb-4">
  <h5 class="fw-semibold mb-3"><i class="bi bi-cpu-fill text-primary me-2"></i>Meet Your AI Companions</h5>
  <div class="row g-3">
    <div class="col-md-6">
      <div class="p-3 rounded-3" style="background:#eff6ff;border-left:4px solid #5b8dee">
        <h6 class="fw-bold text-primary">🎧 Agent 1 – Empathetic Listener</h6>
        <p class="small text-muted mb-0">
          Listens without judgment, detects distress signals, and responds with warmth and active-listening techniques.
          Flags high-risk keywords and triggers emergency protocols automatically.
        </p>
      </div>
    </div>
    <div class="col-md-6">
      <div class="p-3 rounded-3" style="background:#f0fdf4;border-left:4px solid #7ec8a4">
        <h6 class="fw-bold text-success">🌿 Agent 2 – Proactive Care Coach</h6>
        <p class="small text-muted mb-0">
          Suggests personalised mindfulness exercises, breathing routines, and coping strategies based on your
          detected stress level. Always by your side.
        </p>
      </div>
    </div>
  </div>
</div>

<!-- Disclaimer banner -->
<div class="alert alert-warning rounded-3" role="alert">
  <i class="bi bi-exclamation-triangle-fill me-2"></i>
  <strong>Disclaimer:</strong> MindHaven AI is a supportive tool, not a licensed mental health service.
  If you are in crisis, please contact a professional or call an emergency helpline immediately.
  <a href="/help" class="alert-link ms-2">View Helplines →</a>
</div>
""").replace("{% block extra_js %}{% endblock %}", "")


# ── Safe Space Chat template ───────────────────────────────────────────────────
CHAT_HTML = BASE_HTML.replace("{% block title %}Mental Health Companion{% endblock %}", "Safe Space Chat") \
    .replace("{% block content %}{% endblock %}", """
<div class="d-flex align-items-center mb-3 gap-2">
  <h3 class="fw-bold mb-0">Safe Space Chat</h3>
  <span class="badge rounded-pill" style="background:#e0f2fe;color:#0369a1;">
    <i class="bi bi-circle-fill me-1" style="font-size:.5rem"></i>AI Online
  </span>
</div>
<p class="text-muted mb-3 small">
  Share what's on your mind. Both AI agents work together to listen, understand, and support you.
</p>

<div class="mh-card">
  <!-- Chat history box -->
  <div id="chat-box">
    <!-- Welcome message -->
    <div class="d-flex align-items-start gap-2 mb-3">
      <div class="rounded-circle d-flex align-items-center justify-content-center flex-shrink-0"
           style="width:34px;height:34px;background:#dbeafe;">
        <i class="bi bi-heart-pulse-fill text-primary" style="font-size:.85rem"></i>
      </div>
      <div class="chat-bubble bubble-bot">
        <strong>MindHaven AI</strong><br>
        Hello 💙 I'm here with you in this safe space. You can share anything that's on your mind —
        how you're feeling today, what's been weighing on you, or just how your day went.
        There is no judgment here, only care.<br><br>
        <em class="text-muted" style="font-size:.82rem">
          ⚠️ Disclaimer: I am an AI companion, not a licensed therapist.
          This is not a substitute for professional mental health care.
        </em>
      </div>
    </div>
  </div><!-- /chat-box -->

  <!-- Input area -->
  <div class="mt-3 d-flex gap-2">
    <textarea id="user-input"
              class="form-control"
              rows="2"
              placeholder="How are you feeling right now? Type here…"
              style="border-radius:12px;resize:none;"></textarea>
    <button id="send-btn" class="btn btn-primary px-4 rounded-3" onclick="sendMessage()">
      <i class="bi bi-send-fill"></i>
    </button>
  </div>
  <div class="d-flex justify-content-between align-items-center mt-2">
    <span class="text-muted" style="font-size:.78rem">
      <i class="bi bi-lock-fill me-1"></i>Your conversation is private and stays in this session only.
    </span>
    <button class="btn btn-sm btn-outline-secondary rounded-pill" onclick="clearChat()">
      <i class="bi bi-trash me-1"></i>Clear
    </button>
  </div>
</div>
""").replace("{% block extra_js %}{% endblock %}", """
<script>
// ── Chat logic ────────────────────────────────────────────────────────────────
const chatBox   = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn   = document.getElementById('send-btn');

// Submit on Enter (Shift+Enter for newline)
userInput.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

function scrollBottom() { chatBox.scrollTop = chatBox.scrollHeight; }

function appendBubble(html) {
  chatBox.insertAdjacentHTML('beforeend', html);
  scrollBottom();
}

function avatarHtml(initial, bg, color) {
  return `<div class="rounded-circle d-flex align-items-center justify-content-center flex-shrink-0"
              style="width:34px;height:34px;background:${bg};color:${color};font-weight:700;font-size:.85rem">${initial}</div>`;
}

function riskBadge(level) {
  const classes = { low: 'risk-low', moderate: 'risk-moderate', high: 'risk-high' };
  const labels  = { low: '🟢 Low', moderate: '🟡 Moderate', high: '🔴 High Risk' };
  return `<span class="risk-badge ${classes[level]}">${labels[level]}</span>`;
}

function addTypingIndicator() {
  const html = `<div id="typing-indicator" class="d-flex align-items-start gap-2 mb-3">
    <div class="rounded-circle d-flex align-items-center justify-content-center flex-shrink-0"
         style="width:34px;height:34px;background:#dbeafe;">
      <i class="bi bi-heart-pulse-fill text-primary" style="font-size:.85rem"></i>
    </div>
    <div class="chat-bubble bubble-bot typing-dots py-3">
      <span></span><span></span><span></span>
    </div>
  </div>`;
  appendBubble(html);
}

function removeTypingIndicator() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

function renderHelplines(helplines) {
  if (!helplines || helplines.length === 0) return '';
  let rows = helplines.map(h =>
    `<tr><td class="fw-semibold">${h.name}</td><td>${h.number}</td><td class="text-muted small">${h.available}</td></tr>`
  ).join('');
  return `
  <div class="bubble-emergency mt-2">
    <div class="d-flex align-items-center gap-2 mb-2">
      <i class="bi bi-telephone-fill text-danger fs-5"></i>
      <strong class="text-danger">🆘 Immediate Support Available</strong>
    </div>
    <p class="small mb-2">Please reach out to a trained counsellor right now. You are not alone.</p>
    <table class="table table-sm table-borderless mb-0">
      <thead><tr class="text-muted small"><th>Organisation</th><th>Contact</th><th>Hours</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </div>`;
}

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  // Disable input while waiting
  sendBtn.disabled = true;
  userInput.disabled = true;

  // Append user bubble
  appendBubble(`
    <div class="d-flex align-items-start gap-2 mb-3 flex-row-reverse">
      ${avatarHtml('You', '#dbeafe', '#1e40af')}
      <div class="chat-bubble bubble-user">${escapeHtml(text)}</div>
    </div>`);

  userInput.value = '';
  addTypingIndicator();

  try {
    const res  = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    removeTypingIndicator();

    // Agent 1 – Listener reply
    appendBubble(`
      <div class="d-flex align-items-start gap-2 mb-1">
        <div class="rounded-circle d-flex align-items-center justify-content-center flex-shrink-0"
             style="width:34px;height:34px;background:#dbeafe;">
          <i class="bi bi-heart-pulse-fill text-primary" style="font-size:.85rem"></i>
        </div>
        <div>
          <div class="d-flex align-items-center gap-2 mb-1">
            <small class="text-muted fw-semibold">🎧 Empathetic Listener</small>
            ${riskBadge(data.risk_level)}
          </div>
          <div class="chat-bubble bubble-bot">${formatMessage(data.listener_reply)}</div>
        </div>
      </div>`);

    // Agent 2 – Care reply
    appendBubble(`
      <div class="d-flex align-items-start gap-2 mb-3">
        <div class="rounded-circle d-flex align-items-center justify-content-center flex-shrink-0"
             style="width:34px;height:34px;background:#dcfce7;">
          <i class="bi bi-flower1 text-success" style="font-size:.85rem"></i>
        </div>
        <div>
          <small class="text-muted fw-semibold d-block mb-1">🌿 Proactive Care Coach</small>
          <div class="chat-bubble bubble-care">${formatMessage(data.care_reply)}</div>
        </div>
      </div>`);

    // Emergency helplines if high-risk
    if (data.show_emergency) {
      appendBubble(renderHelplines(data.helplines));
    }

  } catch (err) {
    removeTypingIndicator();
    appendBubble(`<div class="alert alert-danger rounded-3">
      Connection error. Please try again or <a href="/help">view helplines</a>.
    </div>`);
  }

  sendBtn.disabled  = false;
  userInput.disabled = false;
  userInput.focus();
}

function clearChat() {
  fetch('/api/clear_chat', { method: 'POST' });
  chatBox.innerHTML = `<div class="text-center text-muted py-4">
    <i class="bi bi-chat-heart display-4 d-block mb-2"></i>Chat cleared. Start fresh 💙
  </div>`;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
          .replace(/"/g,'&quot;').replace(/'/g,'&#039;');
}

// Convert markdown-like **bold** and newlines to HTML
function formatMessage(text) {
  return escapeHtml(text)
    .replace(/[*]{2}(.*?)[*]{2}/g, '<strong>$1</strong>')
    .replace(/[*](.*?)[*]/g, '<em>$1</em>')
    .replace(/\n/g, '<br>');
}
</script>
""")


# ── Coping Resources template ─────────────────────────────────────────────────
RESOURCES_HTML = BASE_HTML.replace("{% block title %}Mental Health Companion{% endblock %}", "Coping Resources") \
    .replace("{% block content %}{% endblock %}", """
<h3 class="fw-bold mb-1">Coping Resources</h3>
<p class="text-muted mb-4">Evidence-based exercises to support your mental wellbeing. Try one right now.</p>

{% for key, res in resources.items() %}
<div class="resource-card">
  <div class="d-flex justify-content-between align-items-start">
    <div>
      <h5 class="fw-semibold mb-1">{{ res.title }}</h5>
      <span class="badge rounded-pill mb-2" style="background:#eff6ff;color:#3b82f6;">{{ res.category }}</span>
      <p class="text-muted mb-1">{{ res.description }}</p>
    </div>
    <span class="badge bg-light text-secondary ms-3 flex-shrink-0">
      <i class="bi bi-clock me-1"></i>{{ res.duration }}
    </span>
  </div>
  <button class="btn btn-sm btn-outline-primary rounded-pill mt-2"
          data-bs-toggle="collapse" data-bs-target="#res-{{ key }}">
    Try This Exercise ↓
  </button>
  <div class="collapse mt-2" id="res-{{ key }}">
    <div class="p-3 rounded-3" style="background:#f8faff;border:1px solid #dbeafe;">
      <p class="mb-0 small">
        <strong>How to practise:</strong> {{ res.description }}<br><br>
        <em class="text-muted">Take your time. There's no right or wrong way. Every effort counts.</em>
      </p>
    </div>
  </div>
</div>
{% endfor %}

<div class="mh-card mt-4">
  <h5 class="fw-semibold mb-2"><i class="bi bi-chat-heart-fill text-primary me-2"></i>Need More Personalised Support?</h5>
  <p class="text-muted mb-3">Our AI Care Coach can suggest exercises tailored to how you're feeling right now.</p>
  <a href="/chat" class="btn btn-primary rounded-pill px-4">Open Safe Space Chat</a>
</div>
""").replace("{% block extra_js %}{% endblock %}", "")


# ── Emergency Help template ───────────────────────────────────────────────────
HELP_HTML = BASE_HTML.replace("{% block title %}Mental Health Companion{% endblock %}", "Get Help Now") \
    .replace("{% block content %}{% endblock %}", """
<div class="text-center mb-4">
  <div class="display-5 mb-2">🆘</div>
  <h2 class="fw-bold text-danger">You Are Not Alone</h2>
  <p class="text-muted">Trained counsellors are available right now. Please reach out.</p>
</div>

<div class="mh-card mb-4" style="border-left:4px solid #f87171;">
  <h5 class="fw-semibold mb-3 text-danger"><i class="bi bi-telephone-fill me-2"></i>Crisis Helplines</h5>
  <div class="table-responsive">
    <table class="table table-hover align-middle mb-0">
      <thead class="table-light">
        <tr>
          <th>Organisation</th>
          <th>Contact</th>
          <th>Availability</th>
        </tr>
      </thead>
      <tbody>
        {% for h in helplines %}
        <tr>
          <td class="fw-semibold">{{ h.name }}</td>
          <td>
            {% if h.number.startswith('http') %}
              <a href="{{ h.number }}" target="_blank" class="text-primary">{{ h.number }}</a>
            {% else %}
              <a href="tel:{{ h.number }}" class="text-danger fw-bold">{{ h.number }}</a>
            {% endif %}
          </td>
          <td class="text-muted small">{{ h.available }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<div class="row g-3 mb-4">
  <div class="col-md-6">
    <div class="mh-card h-100">
      <h6 class="fw-bold text-primary"><i class="bi bi-heart-pulse me-2"></i>In Immediate Danger?</h6>
      <p class="small text-muted">Call your local emergency services immediately:</p>
      <ul class="small text-muted ps-3">
        <li><strong>India:</strong> 112</li>
        <li><strong>United States:</strong> 911</li>
        <li><strong>United Kingdom:</strong> 999</li>
        <li><strong>Australia:</strong> 000</li>
        <li><strong>Europe (general):</strong> 112</li>
      </ul>
    </div>
  </div>
  <div class="col-md-6">
    <div class="mh-card h-100">
      <h6 class="fw-bold text-success"><i class="bi bi-people-fill me-2"></i>Online Communities</h6>
      <ul class="small text-muted ps-3">
        <li><a href="https://www.7cups.com" target="_blank">7 Cups – Free Emotional Support</a></li>
        <li><a href="https://www.reddit.com/r/mentalhealth/" target="_blank">r/mentalhealth – Peer Support</a></li>
        <li><a href="https://www.mind.org.uk" target="_blank">Mind UK – Mental Health Charity</a></li>
        <li><a href="https://nimh.nih.gov" target="_blank">NIMH – Research & Resources</a></li>
      </ul>
    </div>
  </div>
</div>

<div class="alert alert-info">
  <i class="bi bi-info-circle me-2"></i>
  Remember: Seeking help is a sign of strength. You deserve support.
  <a href="/chat" class="alert-link">Talk to our AI companion →</a>
</div>
""").replace("{% block extra_js %}{% endblock %}", "")


# ── About template ────────────────────────────────────────────────────────────
ABOUT_HTML = BASE_HTML.replace("{% block title %}Mental Health Companion{% endblock %}", "About") \
    .replace("{% block content %}{% endblock %}", """
<h3 class="fw-bold mb-1">About MindHaven AI</h3>
<p class="text-muted mb-4">Building compassionate technology for mental health awareness.</p>

<div class="mh-card mb-4">
  <h5 class="fw-semibold mb-2"><i class="bi bi-lightbulb-fill text-warning me-2"></i>Mission</h5>
  <p class="text-muted">
    MindHaven AI is a hackathon project designed to demonstrate how Large Language Models can be
    responsibly applied to mental health support. Our goal is to reduce stigma, provide immediate
    empathetic support, and bridge the gap to professional care.
  </p>
</div>

<div class="mh-card mb-4">
  <h5 class="fw-semibold mb-3"><i class="bi bi-stack text-primary me-2"></i>Technology Stack</h5>
  <div class="row g-3">
    <div class="col-md-6">
      <ul class="list-unstyled small text-muted">
        <li class="mb-2"><i class="bi bi-check-circle-fill text-success me-2"></i><strong>Python + Flask</strong> – Backend framework</li>
        <li class="mb-2"><i class="bi bi-check-circle-fill text-success me-2"></i><strong>Groq API</strong> – Ultra-fast LLM inference</li>
        <li class="mb-2"><i class="bi bi-check-circle-fill text-success me-2"></i><strong>LLaMA 3 8B</strong> – Free open-source LLM</li>
      </ul>
    </div>
    <div class="col-md-6">
      <ul class="list-unstyled small text-muted">
        <li class="mb-2"><i class="bi bi-check-circle-fill text-success me-2"></i><strong>Bootstrap 5</strong> – Responsive UI</li>
        <li class="mb-2"><i class="bi bi-check-circle-fill text-success me-2"></i><strong>Bootstrap Icons</strong> – Accessible iconography</li>
        <li class="mb-2"><i class="bi bi-check-circle-fill text-success me-2"></i><strong>No database</strong> – Session-only, privacy-first</li>
      </ul>
    </div>
  </div>
</div>

<div class="mh-card mb-4">
  <h5 class="fw-semibold mb-2"><i class="bi bi-shield-check-fill text-success me-2"></i>Safety & Ethics</h5>
  <ul class="text-muted small ps-3">
    <li class="mb-1">Keyword-based crisis filter runs <em>before</em> every LLM call.</li>
    <li class="mb-1">High-risk inputs immediately surface verified crisis helplines.</li>
    <li class="mb-1">All responses include mandatory professional-care disclaimers.</li>
    <li class="mb-1">No conversation data is stored — session memory only.</li>
    <li class="mb-1">Open-source model – no proprietary data sharing.</li>
  </ul>
</div>

<div class="alert alert-warning rounded-3">
  <strong><i class="bi bi-exclamation-triangle-fill me-2"></i>Important Disclaimer:</strong>
  MindHaven AI is an <strong>experimental prototype</strong> built for educational and hackathon purposes.
  It is <strong>not</strong> a licensed mental health service and must never be used as a replacement
  for professional diagnosis, therapy, or medical advice.
  In any emergency, call your local crisis line or emergency services immediately.
</div>
""").replace("{% block extra_js %}{% endblock %}", "")


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    """Dashboard / Home page."""
    return render_template_string(HOME_HTML, active_page="home")


@app.route("/chat")
def chat():
    """Safe Space Chat page – initialise history if needed."""
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template_string(CHAT_HTML, active_page="chat")


@app.route("/resources")
def resources():
    """Coping Resources page."""
    return render_template_string(RESOURCES_HTML, active_page="resources", resources=COPING_RESOURCES)


@app.route("/help")
def help_page():
    """Emergency helplines page."""
    return render_template_string(HELP_HTML, active_page="help", helplines=HELPLINES)


@app.route("/about")
def about():
    """About page."""
    return render_template_string(ABOUT_HTML, active_page="about")


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    POST /api/chat
    Body: { "message": "<user text>" }
    Returns orchestrator result as JSON.
    """
    data    = request.get_json(force=True)
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    # Retrieve or initialise session history
    if "chat_history" not in session:
        session["chat_history"] = []

    history = session["chat_history"]

    # ── Run the orchestrator ────────────────────────────────────────────────
    result = orchestrate(message, history)

    # ── Update session history ──────────────────────────────────────────────
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": result["listener_reply"]})
    # Keep history bounded to last 20 messages to avoid token overflow
    session["chat_history"] = history[-20:]
    session.modified = True

    return jsonify(result)


@app.route("/api/clear_chat", methods=["POST"])
def api_clear_chat():
    """Clear the in-session chat history."""
    session.pop("chat_history", None)
    return jsonify({"status": "cleared"})


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("\n⚠️  WARNING: GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
        print("   The app will still run but AI responses will be static fallback messages.\n")
    print("💙 MindHaven AI starting at http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
