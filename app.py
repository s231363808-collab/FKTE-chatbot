import os
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
