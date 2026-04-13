import os
import logging
import time
import hashlib
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# === GEMINI CLIENT SETUP ===
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"  # Best free model for speed + quality

# === CONFIGURATION ===
TEMPERATURE = 0.1      # Low = factual, consistent
MAX_TOKENS = 400       # Keep answers concise
MAX_HISTORY = 4        # Limit conversation turns to save tokens
RATE_LIMIT_WAIT = 5    # Seconds to wait if rate limited
MAX_RETRIES = 2        # Retry attempts for transient errors

# === SYSTEM PROMPT (Strict & Concise) ===
SYSTEM_PROMPT = """
You are UniMAP FKTE academic assistant. Answer ONLY using retrieved context.
Rules:
1. Answer in same language as question (Malay/English)
2. Be concise (max 3 sentences)
3. If unsure: "Sila rujuk FAQ: https://sites.google.com/unimap.edu.my/ur6523003faq/home"
4. NEVER guess policies
5. Cite source when possible

Key info:
- OSI portal: https://osi.unimap.edu.my (login: Matrix Number)
- Drop subject: OSI → Online Forms → Drop Subject Form (Week 4/8 deadlines)
- Max credits: 12-18/semester (GPA≥3.5: petition for 21)
- FKTE contact: ftke@unimap.edu.my
"""

# === KNOWLEDGE BASE (RAG) ===
KNOWLEDGE_BASE = [
    {
        "keywords": ["credit hour", "maksimum kredit", "max credit", "beban kredit", "berapa kredit"],
        "text": "12-18 credit hours/semester. CGPA≥3.5 may petition for 21 credits via OSI → Online Forms → 'Credit Overload Request'.",
        "source": "Academic Guidebook",
        "url": "https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-guide-book"
    },
    {
        "keywords": ["drop subject", "padam subjek", "withdraw", "drop form", "keluar subjek"],
        "text": "OSI → Online Forms → 'Drop Subject Form'. Deadlines: Week 4 (full refund), Week 8 (W grade). After Week 8: Dean approval needed.",
        "source": "OSI Guide",
        "url": "https://osi.unimap.edu.my"
    },
    {
        "keywords": ["register subject", "daftar subjek", "add course", "registration", "tambah subjek"],
        "text": "OSI → Registration → Login with Matrix Number. Check prerequisites & timetable clashes first.",
        "source": "OSI Manual",
        "url": "https://osi.unimap.edu.my/help"
    },
    {
        "keywords": ["deadline", "tarikh", "academic calendar", "kalendar", "important date"],
        "text": "All deadlines in Academic Calendar: https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-calendar",
        "source": "UniMAP Calendar",
        "url": "https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-calendar"
    },
    {
        "keywords": ["contact", "hubungi", "ftke", "email", "advisor", "siapa hubungi"],
        "text": "FKTE: ftke@unimap.edu.my | Office: Level 3, Faculty Building, UniMAP Arau | Phone: +604-979 8000",
        "source": "FKTE Directory",
        "url": "https://ftke.unimap.edu.my/contact"
    }
]

# === PRE-COMPUTED FAQ CACHE (Zero API calls for common questions) ===
FAQ_CACHE = {
    "berapa maximum kredit": "12-18 kredit/semester. CGPA≥3.5 boleh mohon 21 kredit melalui OSI → Online Forms → 'Credit Overload Request'. 📌 Academic Guidebook",
    "how to drop subject": "OSI → Online Forms → 'Drop Subject Form'. Deadline: Week 4 (refund) or Week 8 (W grade). 📌 OSI Guide",
    "how to login osi": "Go to https://osi.unimap.edu.my → Login with Matrix Number and password. 📌 OSI Manual",
    "siapa hubungi fkte": "FKTE: ftke@unimap.edu.my | Office: Level 3, Faculty Building, UniMAP Arau. 📌 FKTE Directory",
    "bila deadline drop": "Week 4 (full refund), Week 8 (W grade). Selepas Week 8 perlu kelulusan Dean. 📌 Academic Calendar",
    "maximum credit hour": "12-18 credits/semester. Students with CGPA ≥ 3.5 may petition for up to 21 credits. 📌 Academic Guidebook",
    "osi login": "https://osi.unimap.edu.my → Login with Matrix Number and password. 📌 OSI Manual",
    "fkte contact": "ftke@unimap.edu.my | Level 3, Faculty Building, UniMAP Arau | +604-979 8000. 📌 FKTE Directory"
}

# === HELPER FUNCTIONS ===
def normalize_query(query):
    """Clean and normalize user query for matching"""
    return query.lower().strip().replace("?", "").replace("!", "").replace(".", "")

def get_faq_answer(query):
    """Check if query matches a pre-computed FAQ (zero API cost)"""
    normalized = normalize_query(query)
    for faq_key, answer in FAQ_CACHE.items():
        if faq_key in normalized or normalized in faq_key:
            return answer
    return None

def retrieve_context(query, top_k=1):
    """Simple keyword retrieval - returns MOST relevant chunk only"""
    query_lower = query.lower()
    best_score = 0
    best_chunk = None
    for chunk in KNOWLEDGE_BASE:
        score = sum(1 for kw in chunk["keywords"] if kw.lower() in query_lower)
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return [best_chunk] if best_score > 0 and best_chunk else []

def format_context(chunks):
    """Format context concisely"""
    if not chunks:
        return ""
    c = chunks[0]
    return f"📌 {c['source']}: {c['text']}"

def call_gemini_with_retry(prompt, max_retries=MAX_RETRIES):
    """Call Gemini API with retry logic for rate limits"""
    model = genai.GenerativeModel(MODEL_NAME)
    generation_config = genai.types.GenerationConfig(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
        top_p=0.9,
    )
    
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(prompt, generation_config=generation_config)
            return response.text
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                if attempt < max_retries:
                    wait_time = RATE_LIMIT_WAIT * (attempt + 1)
                    logger.warning(f"Gemini rate limited. Waiting {wait_time}s... (attempt {attempt+1})")
                    time.sleep(wait_time)
                    continue
            # Re-raise if not rate limit or max retries reached
            raise

# In-memory cache for recent answers (resets on server restart)
response_cache = {}
CACHE_TTL = 300  # 5 minutes

def get_cached_response(query):
    """Get response from cache if recent"""
    key = hashlib.md5(query.lower().encode()).hexdigest()
    cached = response_cache.get(key)
    if cached:
        reply, timestamp = cached
        if time.time() - timestamp < CACHE_TTL:
            return reply
    return None

def save_to_cache(query, reply):
    """Save response to cache"""
    key = hashlib.md5(query.lower().encode()).hexdigest()
    response_cache[key] = (reply, time.time())

# Conversation history storage (for context awareness)
conversations = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True)
        if not 
            return jsonify({"error": "Invalid request"}), 400
        
        session_id = data.get("session_id", "default")
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        logger.info(f"[{session_id}] Q: {user_message[:60]}...")

        # 🎯 STRATEGY 1: Check FAQ cache first (ZERO API cost)
        faq_answer = get_faq_answer(user_message)
        if faq_answer:
            logger.info(f"[{session_id}] ✅ FAQ cache hit")
            save_to_cache(user_message, faq_answer)
            return jsonify({"reply": faq_answer, "cached": True, "source": "faq"})

        # 🎯 STRATEGY 2: Check recent response cache
        cached = get_cached_response(user_message)
        if cached:
            logger.info(f"[{session_id}] ✅ Response cache hit")
            return jsonify({"reply": cached, "cached": True, "source": "cache"})

        # Initialize conversation history
        if session_id not in conversations:
            conversations[session_id] = []

        # 🎯 STRATEGY 3: RAG - Get relevant policy context
        context_chunks = retrieve_context(user_message)
        context_text = format_context(context_chunks)

        # Build minimal, efficient prompt for Gemini
        if context_text:
            prompt = f"""Policy Context:
{context_text}

Student Question: {user_message}

Answer concisely based ONLY on the context above. 
If unsure, direct to FAQ Centre: https://sites.google.com/unimap.edu.my/ur6523003faq/home
Answer in the same language as the question (Malay/English). Be helpful and polite."""
        else:
            prompt = f"""Question: {user_message}

If you don't have specific policy information, politely direct the student to:
- FAQ Centre: https://sites.google.com/unimap.edu.my/ur6523003faq/home
- FKTE Contact: ftke@unimap.edu.my

Answer in the same language as the question. Keep it short and helpful."""

        # 🎯 STRATEGY 4: Call Gemini API with retry
        try:
            bot_reply = call_gemini_with_retry(prompt)
            bot_reply = bot_reply.strip()
        except Exception as e:
            logger.warning(f"Gemini API failed: {str(e)[:100]}")
            # 🎯 STRATEGY 5: Fallback when API fails
            if context_chunks:
                bot_reply = f"📌 {context_chunks[0]['source']}: {context_chunks[0]['text']}\n\n⚠️ *Untuk pengesahan rasmi, sila rujuk FAQ Centre*"
            else:
                bot_reply = "Maaf, sistem sedang sibuk. Sila rujuk Pusat FAQ: https://sites.google.com/unimap.edu.my/ur6523003faq/home atau hubungi ftke@unimap.edu.my"

        # 🎯 STRATEGY 6: Cache successful non-fallback responses
        if "faq" not in bot_reply.lower() and "maaf" not in bot_reply.lower() and "sibuk" not in bot_reply.lower():
            save_to_cache(user_message, bot_reply)

        # Save to conversation history (for multi-turn context)
        conversations[session_id].append({"role": "user", "content": user_message})
        conversations[session_id].append({"role": "assistant", "content": bot_reply})
        conversations[session_id] = conversations[session_id][-MAX_HISTORY:]

        logger.info(f"[{session_id}] A: {bot_reply[:60]}...")
        return jsonify({"reply": bot_reply, "source": "gemini"})

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        fallback = "Maaf, terdapat masalah teknikal. Sila cuba sebentar lagi atau rujuk FAQ: https://sites.google.com/unimap.edu.my/ur6523003faq/home"
        return jsonify({"reply": fallback, "error": True}), 200

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok", 
        "service": "unimap-chatbot-gemini",
        "model": MODEL_NAME,
        "provider": "google-gemini-free"
    })

@app.route("/clear/<session_id>", methods=["POST"])
def clear_session(session_id):
    """Clear conversation history and cache for a session"""
    if session_id in conversations:
        del conversations[session_id]
    # Clear cached responses for this session's queries
    keys_to_remove = [k for k in response_cache if session_id in k]
    for k in keys_to_remove:
        del response_cache[k]
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    logger.info(f"🚀 UniMAP Chatbot (Gemini Free) on port {port} | Model: {MODEL_NAME}")
    app.run(debug=debug, host="0.0.0.0", port=port)import os
import logging
import time
import hashlib
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# OpenRouter client (FREE TIER)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

# === CONFIGURATION FOR FREE TIER ===
MODEL_NAME = "openrouter/free"
TEMPERATURE = 0.1  # Low = consistent, factual
MAX_TOKENS = 300   # Short answers = fewer tokens = more requests
MAX_HISTORY = 4    # Keep conversation short to save tokens
RATE_LIMIT_WAIT = 5  # Seconds to wait when rate limited
MAX_RETRIES = 2    # Only retry once to avoid long waits

# === SYSTEM PROMPT (Strict & Concise) ===
SYSTEM_PROMPT = """
You are UniMAP FKTE academic assistant. Answer ONLY using retrieved context.
Rules:
1. Answer in same language as question (Malay/English)
2. Be concise (max 3 sentences)
3. If unsure: "Sila rujuk FAQ: https://sites.google.com/unimap.edu.my/ur6523003faq/home"
4. NEVER guess policies
5. Cite source when possible

Key info:
- OSI portal: https://osi.unimap.edu.my (login: Matrix Number)
- Drop subject: OSI → Online Forms → Drop Subject Form (Week 4/8 deadlines)
- Max credits: 12-18/semester (GPA≥3.5: petition for 21)
- FKTE contact: ftke@unimap.edu.my
"""

# === KNOWLEDGE BASE (RAG) ===
KNOWLEDGE_BASE = [
    {
        "keywords": ["credit hour", "maksimum kredit", "max credit", "beban kredit", "berapa kredit"],
        "text": "12-18 credit hours/semester. CGPA≥3.5 may petition for 21 credits via OSI → Online Forms → 'Credit Overload Request'.",
        "source": "Academic Guidebook",
        "url": "https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-guide-book"
    },
    {
        "keywords": ["drop subject", "padam subjek", "withdraw", "drop form", "keluar subjek"],
        "text": "OSI → Online Forms → 'Drop Subject Form'. Deadlines: Week 4 (full refund), Week 8 (W grade). After Week 8: Dean approval needed.",
        "source": "OSI Guide",
        "url": "https://osi.unimap.edu.my"
    },
    {
        "keywords": ["register subject", "daftar subjek", "add course", "registration", "tambah subjek"],
        "text": "OSI → Registration → Login with Matrix Number. Check prerequisites & timetable clashes first.",
        "source": "OSI Manual",
        "url": "https://osi.unimap.edu.my/help"
    },
    {
        "keywords": ["deadline", "tarikh", "academic calendar", "kalendar", "important date"],
        "text": "All deadlines in Academic Calendar: https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-calendar",
        "source": "UniMAP Calendar",
        "url": "https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-calendar"
    },
    {
        "keywords": ["contact", "hubungi", "ftke", "email", "advisor", "siapa hubungi"],
        "text": "FKTE: ftke@unimap.edu.my | Office: Level 3, Faculty Building, UniMAP Arau | Phone: +604-979 8000",
        "source": "FKTE Directory",
        "url": "https://ftke.unimap.edu.my/contact"
    }
]

# === PRE-COMPUTED FAQ CACHE (Zero API calls for common questions) ===
FAQ_CACHE = {
    "berapa maximum kredit": "12-18 kredit/semester. CGPA≥3.5 boleh mohon 21 kredit melalui OSI → Online Forms → 'Credit Overload Request'. 📌 Academic Guidebook",
    "how to drop subject": "OSI → Online Forms → 'Drop Subject Form'. Deadline: Week 4 (refund) or Week 8 (W grade). 📌 OSI Guide",
    "how to login osi": "Go to https://osi.unimap.edu.my → Login with Matrix Number and password. 📌 OSI Manual",
    "siapa hubungi fkte": "FKTE: ftke@unimap.edu.my | Office: Level 3, Faculty Building, UniMAP Arau. 📌 FKTE Directory",
    "bila deadline drop": "Week 4 (full refund), Week 8 (W grade). Selepas Week 8 perlu kelulusan Dean. 📌 Academic Calendar",
    "maximum credit hour": "12-18 credits/semester. Students with CGPA ≥ 3.5 may petition for up to 21 credits. 📌 Academic Guidebook",
    "osi login": "https://osi.unimap.edu.my → Login with Matrix Number and password. 📌 OSI Manual",
    "fkte contact": "ftke@unimap.edu.my | Level 3, Faculty Building, UniMAP Arau | +604-979 8000. 📌 FKTE Directory"
}

# === HELPER FUNCTIONS ===
def normalize_query(query):
    """Clean and normalize user query for matching"""
    return query.lower().strip().replace("?", "").replace("!", "").replace(".", "")

def get_faq_answer(query):
    """Check if query matches a pre-computed FAQ (zero API cost)"""
    normalized = normalize_query(query)
    for faq_key, answer in FAQ_CACHE.items():
        if faq_key in normalized or normalized in faq_key:
            return answer
    return None

def retrieve_context(query, top_k=1):
    """Simple keyword retrieval - returns MOST relevant chunk only"""
    query_lower = query.lower()
    best_score = 0
    best_chunk = None
    for chunk in KNOWLEDGE_BASE:
        score = sum(1 for kw in chunk["keywords"] if kw.lower() in query_lower)
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return [best_chunk] if best_score > 0 and best_chunk else []

def format_context(chunks):
    """Format context concisely"""
    if not chunks:
        return ""
    c = chunks[0]
    return f"📌 {c['source']}: {c['text']}"

def call_api_with_retry(messages):
    """Call API with minimal retry for rate limits"""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                top_p=0.9,
                extra_headers={
                    "HTTP-Referer": "https://ftke.unimap.edu.my",
                    "X-Title": "UniMAP FKTE Assistant"
                }
            )
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < MAX_RETRIES:
                    logger.warning(f"Rate limited. Waiting {RATE_LIMIT_WAIT}s...")
                    time.sleep(RATE_LIMIT_WAIT)
                    continue
            raise  # Re-raise if not rate limit or max retries reached

# In-memory cache for recent answers (resets on server restart)
response_cache = {}
CACHE_TTL = 300  # 5 minutes

def get_cached_response(query):
    """Get response from cache if recent"""
    key = hashlib.md5(query.lower().encode()).hexdigest()
    cached = response_cache.get(key)
    if cached:
        reply, timestamp = cached
        if time.time() - timestamp < CACHE_TTL:
            return reply
    return None

def save_to_cache(query, reply):
    """Save response to cache"""
    key = hashlib.md5(query.lower().encode()).hexdigest()
    response_cache[key] = (reply, time.time())

# Conversation history storage
conversations = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True)
        if not 
            return jsonify({"error": "Invalid request"}), 400
        
        session_id = data.get("session_id", "default")
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        logger.info(f"[{session_id}] Q: {user_message[:60]}...")

        # 🎯 STRATEGY 1: Check FAQ cache first (ZERO API cost)
        faq_answer = get_faq_answer(user_message)
        if faq_answer:
            logger.info(f"[{session_id}] ✅ FAQ cache hit")
            save_to_cache(user_message, faq_answer)
            return jsonify({"reply": faq_answer, "cached": True})

        # 🎯 STRATEGY 2: Check recent response cache
        cached = get_cached_response(user_message)
        if cached:
            logger.info(f"[{session_id}] ✅ Response cache hit")
            return jsonify({"reply": cached, "cached": True})

        # Initialize conversation history
        if session_id not in conversations:
            conversations[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 🎯 STRATEGY 3: RAG - Get relevant policy context
        context_chunks = retrieve_context(user_message)
        context_text = format_context(context_chunks)

        # Build minimal prompt
        if context_text:
            user_content = f"Policy: {context_text}\n\nQuestion: {user_message}\n\nAnswer concisely in same language."
        else:
            user_content = f"Question: {user_message}\n\nIf unsure, direct to FAQ Centre."

        # Add to history (keep it short)
        conversations[session_id].append({"role": "user", "content": user_content})
        conversations[session_id] = conversations[session_id][-MAX_HISTORY:]

        # 🎯 STRATEGY 4: Call API with retry
        try:
            response = call_api_with_retry(conversations[session_id])
            bot_reply = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"API failed: {str(e)[:100]}")
            # 🎯 STRATEGY 5: Fallback when API fails
            if context_chunks:
                bot_reply = f"📌 {context_chunks[0]['source']}: {context_chunks[0]['text']}\n\n⚠️ *Untuk pengesahan rasmi, sila rujuk FAQ Centre*"
            else:
                bot_reply = "Maaf, sistem sedang sibuk. Sila rujuk Pusat FAQ: https://sites.google.com/unimap.edu.my/ur6523003faq/home atau hubungi ftke@unimap.edu.my"

        # 🎯 STRATEGY 6: Cache successful responses
        if "faq" not in bot_reply.lower() and "maaf" not in bot_reply.lower():
            save_to_cache(user_message, bot_reply)

        # Save to conversation history (for context)
        conversations[session_id].append({"role": "assistant", "content": bot_reply})

        logger.info(f"[{session_id}] A: {bot_reply[:60]}...")
        return jsonify({"reply": bot_reply})

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        fallback = "Maaf, terdapat masalah teknikal. Sila cuba sebentar lagi atau rujuk FAQ: https://sites.google.com/unimap.edu.my/ur6523003faq/home"
        return jsonify({"reply": fallback, "error": True}), 200

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "unimap-chatbot-free"})

@app.route("/clear/<session_id>", methods=["POST"])
def clear_session(session_id):
    """Clear conversation history"""
    if session_id in conversations:
        del conversations[session_id]
    if session_id in response_cache:
        del response_cache[session_id]
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    logger.info(f"🚀 UniMAP Chatbot (FREE) on port {port}")
    app.run(debug=debug, host="0.0.0.0", port=port)import os
import logging
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# OpenRouter client (FREE TIER)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

# === SYSTEM PROMPT ===
SYSTEM_PROMPT = """
You are a helpful university assistant chatbot for UniMAP (Universiti Malaysia Perlis).
You specifically help students from the Faculty of Electrical Engineering & Technology (FKTE).

PROGRAMMES YOU SUPPORT:
- Bachelor of Mechatronic Engineering with Honours (UR6523003)
- Bachelor of Electrical Engineering with Honours (UR6522001)
- Bachelor of Electrical Engineering Technology (Industrial Power) with Honours (UR6522002)
- Bachelor of Electrical Engineering Technology (Robotics and Automation) with Honours (UR6523006)
- Bachelor of Technology in Electrical System Maintenance with Honours (UR6712001)
- Diploma in Electrical Engineering (UR4522001)
- Diploma in Mechatronic Engineering (UR4523001)

YOU HELP WITH:
1. Course registration, adding/dropping subjects
2. Academic forms and where to find them
3. Curriculum structure and course requirements
4. Academic rules and policies
5. Important deadlines
6. Who to contact for help

IMPORTANT UNIMAP INFO:
- Student portal: OSI at https://osi.unimap.edu.my (login with Matrix Number)
- To register: OSI → Registration → Login
- To drop subject: OSI → Online Forms → Drop Subject Form
- FKTE contact: ftke@unimap.edu.my
- Main campus: Arau, Perlis

REFERENCE LINKS (share when relevant):
- UniMAP: https://www.unimap.edu.my/
- FKTE: https://ftke.unimap.edu.my/
- Academic Calendar: https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-calendar
- Academic Guidebook: https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-guide-book
- Class Timetable: https://sites.google.com/unimap.edu.my/academicunimap/class-timetable
- FAQ Centre: https://sites.google.com/unimap.edu.my/ur6523003faq/home

GUIDELINES:
- Answer in the SAME language the student uses (Malay or English)
- Be friendly, polite, and concise
- If you don't know, say: "Sila rujuk Pusat FAQ: https://sites.google.com/unimap.edu.my/ur6523003faq/home"
- NEVER guess policies. Accuracy is critical.
"""

# === KNOWLEDGE BASE (RAG) ===
KNOWLEDGE_BASE = [
    {
        "keywords": ["credit hour", "maksimum kredit", "max credit", "beban kredit"],
        "text": "Undergraduate students: 12-18 credit hours/semester. CGPA ≥ 3.5 may petition for up to 21 credits via OSI → Online Forms → 'Credit Overload Request'.",
        "source": "Academic Guidebook 2024",
        "url": "https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-guide-book"
    },
    {
        "keywords": ["drop subject", "padam subjek", "withdraw", "drop form"],
        "text": "Drop subject: OSI → Online Forms → 'Drop Subject Form'. Deadlines: Week 4 (full refund), Week 8 (W grade). After Week 8 needs Dean approval.",
        "source": "OSI Guide",
        "url": "https://osi.unimap.edu.my"
    },
    {
        "keywords": ["register subject", "daftar subjek", "add course", "registration"],
        "text": "Register via OSI: https://osi.unimap.edu.my → Registration → Login with Matrix Number. Check prerequisites and timetable clashes first.",
        "source": "OSI Manual",
        "url": "https://osi.unimap.edu.my/help"
    },
    {
        "keywords": ["deadline", "tarikh", "academic calendar", "kalendar"],
        "text": "All academic deadlines are in the Academic Calendar. Check it yearly: https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-calendar",
        "source": "UniMAP Academic Calendar",
        "url": "https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-calendar"
    },
    {
        "keywords": ["contact", "hubungi", "ftke", "email", "advisor"],
        "text": "FKTE academic matters: ftke@unimap.edu.my | Office: Level 3, Faculty Building, UniMAP Arau | Phone: +604-979 8000",
        "source": "FKTE Directory",
        "url": "https://ftke.unimap.edu.my/contact"
    }
]

# === HELPER FUNCTIONS ===
MAX_HISTORY = 6

def retrieve_context(query):
    """Simple keyword search - returns relevant policy chunks"""
    query_lower = query.lower()
    results = []
    for chunk in KNOWLEDGE_BASE:
        score = sum(1 for kw in chunk["keywords"] if kw.lower() in query_lower)
        if score > 0:
            results.append((score, chunk))
    results.sort(reverse=True)
    return [chunk for score, chunk in results[:2]]

def format_context(chunks):
    """Turn retrieved chunks into readable text for the AI"""
    if not chunks:
        return "No specific policy found in knowledge base."
    output = []
    for c in chunks:
        output.append(f"📌 Source: {c['source']}\n🔗 {c['url']}\nℹ️ {c['text']}")
    return "\n\n".join(output)

# Store conversation history
conversations = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # ✅ FIX 1: Properly get and validate JSON data
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid request or missing JSON"}), 400
        
        session_id = data.get("session_id", "default")
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        logger.info(f"[{session_id}] User: {user_message[:80]}...")

        # Initialize conversation with system prompt
        if session_id not in conversations:
            conversations[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 🔍 RAG: Retrieve relevant policy context
        context_chunks = retrieve_context(user_message)
        context_text = format_context(context_chunks)

        # Build augmented message
        augmented_user_msg = f"""
📚 Retrieved Policy Context:
{context_text}

❓ Student Question: {user_message}

✅ Instructions:
- Answer based ONLY on the context above and your system prompt.
- If context is missing, direct student to FAQ Centre or ftke@unimap.edu.my
- Answer in the same language as the question (Malay/English)
- Be concise and cite sources when possible.
"""

        # Add to history and trim to prevent token overflow
        conversations[session_id].append({"role": "user", "content": augmented_user_msg})
        conversations[session_id] = conversations[session_id][-MAX_HISTORY:]

        # 🤖 Call OpenRouter FREE model
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=conversations[session_id],
            temperature=0.2,
            max_tokens=400,
            top_p=0.9,
            extra_headers={
                "HTTP-Referer": "https://ftke.unimap.edu.my",
                "X-Title": "UniMAP FKTE Assistant"
            }
        )

        bot_reply = response.choices[0].message.content.strip()

        # 🛡️ Add fallback if no context was found
        if not context_chunks:
            bot_reply += "\n\n⚠️ *Maklumat ini berdasarkan panduan umum. Sila sahkan di Pusat FAQ: https://sites.google.com/unimap.edu.my/ur6523003faq/home*"

        conversations[session_id].append({"role": "assistant", "content": bot_reply})
        logger.info(f"[{session_id}] Bot reply sent ({len(bot_reply)} chars)")
        
        return jsonify({"reply": bot_reply})

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        fallback = "Maaf, terdapat masalah teknikal. Sila cuba sebentar lagi atau hubungi ftke@unimap.edu.my."
        return jsonify({"reply": fallback, "error": True}), 200

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "unimap-chatbot"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    logger.info(f"🚀 Starting UniMAP Chatbot on port {port} (debug={debug})")
    app.run(debug=debug, host="0.0.0.0", port=port)
