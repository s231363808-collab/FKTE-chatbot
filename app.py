import os
import logging
import time
import hashlib
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"

TEMPERATURE = 0.1
MAX_TOKENS = 1500
MAX_HISTORY = 6
RATE_LIMIT_WAIT = 5
MAX_RETRIES = 2

SYSTEM_PROMPT = """
You help students from ALL programmes under FKTE (Faculty of Electrical Engineering & Technology):

DEGREE PROGRAMMES:
- Bachelor of Mechatronic Engineering with Honours (UR6523003)
- Bachelor of Electrical Engineering with Honours (UR6522001)
- Bachelor of Electrical Engineering Technology (Industrial Power) with Honours (UR6522002)
- Bachelor of Electrical Engineering Technology (Robotics and Automation Technology) with Honours (UR6523006)
- Bachelor of Technology in Electrical System Maintenance with Honours (UR6712001)

DIPLOMA PROGRAMMES:
- Diploma in Electrical Engineering (UR4522001)
- Diploma in Mechatronic Engineering (UR4523001)

CRITICAL RULES:
- Student portal is OSI at https://osi.unimap.edu.my — NEVER say i-Maaluum
- Login to OSI using Matrix Number and password
- Answer in same language as student (Malay or English)
- Always provide relevant links
- If unsure, refer to FAQ Centre or relevant contact person

KEY CONTACTS:
- Head of Department: Assoc. Prof. Dr. Kamarulzaman Kamarudin | kamarulzaman@unimap.edu.my | WhatsApp: https://wa.me/60142307071
- Programme Chairman (PR): Assoc. Prof. Dr. Abdul Halim Ismail | ihalim@unimap.edu.my | WhatsApp: https://wa.me/60124542662
- Assistant Registrar: Mdm. Hanimah Karjoo | hanimah@unimap.edu.my | WhatsApp: https://wa.me/601137243477
- Assistant Registrar: Mdm. Salwana Hafizah | salwanahafizah@unimap.edu.my | WhatsApp: https://wa.me/60126501559
- FYP Coordinator: Dr. Hassrizal Hassan Basri | hassrizal@unimap.edu.my | WhatsApp: https://wa.me/601137007588
- IDP Coordinator: Assoc. Prof. Dr. Muhammad Khairul Ali Hassan | khairulhassan@unimap.edu.my | WhatsApp: https://wa.me/60124226670

ACADEMIC FORMS (all at https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms):
- HEA(B)-01[a]: Borang Permohonan Pemindahan Kredit Vertikal
- HEA(B)-01[b]: Borang Permohonan Pemindahan Kredit Horizontal
- HEA(B)-02[a]: Borang Pendaftaran Kursus Status Percubaan
- HEA(B)-02[b]: Borang Pendaftaran Kursus Lewat (NEW)
- HEA(B)-03: Borang Pendaftaran Gugur Kursus (Drop Subject)
- HEA(B)-04: Borang Tarik Diri Kursus (Withdraw)
- HEA(B)-06: Borang Permohonan Pertukaran Program Pengajian (NEW)
- HEA(B)-07: Borang Permohonan Tangguh Pengajian (Deferment) (NEW)
- HEA(B)-08: Borang Permohonan Berhenti Daripada Pengajian
- HEA(B)-09: Borang Penamatan Pengajian Pelajar
- HEA(B)-10: Borang Rayuan Kemasukan Semula (NEW)

CURRICULUM STRUCTURE (by intake year):
- 2020/2021: Matric starts with 20106xxxx
- 2021/2022: Matric starts with 21106xxxx
- 2022/2023: Matric starts with 22106xxxx
- 2023/2024: Matric starts with 23106xxxx
- 2024/2025: Matric starts with 24106xxxx
- 2025/2026: Matric starts with 25106xxxx
Full details: https://sites.google.com/unimap.edu.my/ur6523003faq/curriculum-structure

ACADEMIC CALENDAR:
- Bachelor 2025/2026: https://www.unimap.edu.my/images/calendar/KALENDAR_AKADEMIK-SARJANA MUDA_UNIMAP_ 20252026-07012026.pdf
- Bachelor 2026/2027: https://www.unimap.edu.my/images/calendar/KALENDAR-AKADEMIK-SARJANA MUDA_20262027.pdf
- All calendars: https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-calendar
- Calendar queries: Shazlina Isakh | shazlina@unimap.edu.my | Tel: +604 941 4060

REFERENCE LINKS:
- FAQ Centre (Mechatronic): https://sites.google.com/unimap.edu.my/ur6523003faq/home
- OSI Portal: https://osi.unimap.edu.my
- UniMAP Website: https://www.unimap.edu.my/
- FKTE Website: https://ftke.unimap.edu.my/
- FKTE Student Site: https://sites.google.com/unimap.edu.my/fkte-undergraduate
- Academic Guidebook: https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-guide-book
- Academic Regulation: https://drive.google.com/file/d/1FaoqxqhYUEn9eHe0ZlBkW_houxgapvtq/view
- Academic Regulation (2025): https://drive.google.com/file/d/1e8ZSr3-khfBDd7LEYUimTcU0vbbjxQf3/view
- Class Timetable: https://sites.google.com/unimap.edu.my/academicunimap/class-timetable
- Student Dress Code: https://www.unimap.edu.my/index.php/en/campus-life/reference/student-dress-code
"""

KNOWLEDGE_BASE = [
    {
        "keywords": ["drop subject", "gugur kursus", "padam subjek", "drop course", "keluar subjek", "borang gugur"],
        "text": "To drop a subject: Download form HEA(B)-03 from https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms and submit to Assistant Registrar Mdm. Hanimah (hanimah@unimap.edu.my / wa.me/601137243477).",
        "source": "Academic Forms",
        "url": "https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms"
    },
    {
        "keywords": ["register subject", "daftar subjek", "tambah subjek", "add subject", "course registration", "pendaftaran kursus", "late registration", "lambat daftar"],
        "text": "Log in to OSI portal at https://osi.unimap.edu.my using your Matrix Number and password to register subjects. For late registration use form HEA(B)-02[b].",
        "source": "OSI Portal",
        "url": "https://osi.unimap.edu.my"
    },
    {
        "keywords": ["withdraw", "tarik diri", "withdrawal"],
        "text": "To withdraw from a course, download form HEA(B)-04 Borang Tarik Diri Kursus from: https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms",
        "source": "Academic Forms",
        "url": "https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms"
    },
    {
        "keywords": ["defer", "tangguh", "tangguh pengajian", "postpone", "deferment"],
        "text": "To defer studies, download form HEA(B)-07 Borang Permohonan Tangguh Pengajian from: https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms",
        "source": "Academic Forms",
        "url": "https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms"
    },
    {
        "keywords": ["curriculum", "course structure", "struktur kursus", "what subject", "intake", "matric"],
        "text": "Curriculum depends on intake year. Check matric prefix: 20106=2020/21, 21106=2021/22, 22106=2022/23, 23106=2023/24, 24106=2024/25, 25106=2025/26. Details: https://sites.google.com/unimap.edu.my/ur6523003faq/curriculum-structure",
        "source": "Curriculum Structure",
        "url": "https://sites.google.com/unimap.edu.my/ur6523003faq/curriculum-structure"
    },
    {
        "keywords": ["contact", "hubungi", "siapa", "who", "email", "whatsapp", "phone", "telefon"],
        "text": "Programme Chairman: Dr. Abdul Halim (ihalim@unimap.edu.my, wa.me/60124542662). Assistant Registrar: Mdm. Hanimah (hanimah@unimap.edu.my, wa.me/601137243477). Head of Dept: Dr. Kamarulzaman (kamarulzaman@unimap.edu.my, wa.me/60142307071).",
        "source": "Contact Us",
        "url": "https://sites.google.com/unimap.edu.my/ur6523003faq/contact-us"
    },
    {
        "keywords": ["deadline", "tarikh", "calendar", "kalendar", "important date", "tarikh penting"],
        "text": "Academic Calendar: https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-calendar. For queries contact Shazlina Isakh: shazlina@unimap.edu.my | Tel: +604 941 4060",
        "source": "Academic Calendar",
        "url": "https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-calendar"
    },
    {
        "keywords": ["fyp", "final year project", "projek tahun akhir"],
        "text": "FYP Coordinator: Dr. Hassrizal Hassan Basri | hassrizal@unimap.edu.my | WhatsApp: https://wa.me/601137007588",
        "source": "Contact Us",
        "url": "https://sites.google.com/unimap.edu.my/ur6523003faq/contact-us"
    },
    {
        "keywords": ["idp", "integrated design project"],
        "text": "IDP Coordinator: Assoc. Prof. Dr. Muhammad Khairul Ali Hassan | khairulhassan@unimap.edu.my | WhatsApp: https://wa.me/60124226670",
        "source": "Contact Us",
        "url": "https://sites.google.com/unimap.edu.my/ur6523003faq/contact-us"
    },
    {
        "keywords": ["transfer program", "tukar program", "pertukaran program", "change programme"],
        "text": "To transfer programme, download form HEA(B)-06 from: https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms",
        "source": "Academic Forms",
        "url": "https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms"
    },
    {
        "keywords": ["osi", "student portal", "portal pelajar", "login", "log in", "sign in"],
        "text": "Student portal is OSI at https://osi.unimap.edu.my. Login with Matrix Number and password.",
        "source": "OSI Portal",
        "url": "https://osi.unimap.edu.my"
    },
    {
        "keywords": ["credit transfer", "pindahan kredit", "transfer kredit"],
        "text": "Credit transfer: Vertical use HEA(B)-01[a], Horizontal use HEA(B)-01[b]. Download from: https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms",
        "source": "Academic Forms",
        "url": "https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms"
    },
    {
        "keywords": ["timetable", "jadual", "class schedule", "jadual kuliah"],
        "text": "Class timetable available at: https://sites.google.com/unimap.edu.my/academicunimap/class-timetable",
        "source": "Class Timetable",
        "url": "https://sites.google.com/unimap.edu.my/academicunimap/class-timetable"
    },
    {
        "keywords": ["dress code", "pakaian", "attire", "uniform"],
        "text": "Student dress code: https://www.unimap.edu.my/index.php/en/campus-life/reference/student-dress-code",
        "source": "Student Dress Code",
        "url": "https://www.unimap.edu.my/index.php/en/campus-life/reference/student-dress-code"
    },
    {
        "keywords": ["guidebook", "buku panduan", "academic guide", "regulation", "peraturan"],
        "text": "Academic Guidebook: https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-guide-book | Academic Regulation (2025): https://drive.google.com/file/d/1e8ZSr3-khfBDd7LEYUimTcU0vbbjxQf3/view",
        "source": "Academic Guidebook",
        "url": "https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-guide-book"
    },
    {
        "keywords": ["stop study", "berhenti", "quit", "keluar universiti", "terminate"],
        "text": "To stop studies: HEA(B)-08 Borang Permohonan Berhenti. To terminate: HEA(B)-09 Borang Penamatan. Download from: https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms",
        "source": "Academic Forms",
        "url": "https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms"
    },
    {
        "keywords": ["readmission", "rayuan masuk semula", "reinstatement"],
        "text": "For readmission, use form HEA(B)-10 Borang Rayuan Kemasukan Semula from: https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms",
        "source": "Academic Forms",
        "url": "https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms"
    }
]

FAQ_CACHE = {
    "how to drop subject": "Download form HEA(B)-03 (Borang Pendaftaran Gugur Kursus) from https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms and submit to Mdm. Hanimah (hanimah@unimap.edu.my).",
    "cara gugur subjek": "Muat turun borang HEA(B)-03 dari https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms dan hantar kepada Mdm. Hanimah (hanimah@unimap.edu.my).",
    "how to register subject": "Log in to OSI portal at https://osi.unimap.edu.my using your Matrix Number and password.",
    "cara daftar subjek": "Log masuk ke portal OSI di https://osi.unimap.edu.my menggunakan Nombor Matrik dan kata laluan.",
    "student portal": "Student portal is OSI at https://osi.unimap.edu.my. Login with Matrix Number and password.",
    "portal pelajar": "Portal pelajar adalah OSI di https://osi.unimap.edu.my. Log masuk dengan Nombor Matrik dan kata laluan.",
    "contact programme chairman": "Assoc. Prof. Dr. Abdul Halim Ismail | ihalim@unimap.edu.my | WhatsApp: https://wa.me/60124542662",
    "siapa pengerusi program": "Assoc. Prof. Dr. Abdul Halim Ismail | ihalim@unimap.edu.my | WhatsApp: https://wa.me/60124542662",
    "contact head of department": "Assoc. Prof. Dr. Kamarulzaman Kamarudin | kamarulzaman@unimap.edu.my | WhatsApp: https://wa.me/60142307071",
    "ketua jabatan": "Assoc. Prof. Dr. Kamarulzaman Kamarudin | kamarulzaman@unimap.edu.my | WhatsApp: https://wa.me/60142307071",
    "academic forms": "All academic forms: https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms",
    "borang akademik": "Semua borang akademik: https://sites.google.com/unimap.edu.my/ur6523003faq/academic-forms",
    "curriculum structure": "Curriculum by intake: https://sites.google.com/unimap.edu.my/ur6523003faq/curriculum-structure",
    "struktur kurikulum": "Struktur kurikulum: https://sites.google.com/unimap.edu.my/ur6523003faq/curriculum-structure",
    "fyp coordinator": "Dr. Hassrizal Hassan Basri | hassrizal@unimap.edu.my | WhatsApp: https://wa.me/601137007588",
    "idp coordinator": "Assoc. Prof. Dr. Muhammad Khairul Ali Hassan | khairulhassan@unimap.edu.my | WhatsApp: https://wa.me/60124226670",
    "class timetable": "Class timetable: https://sites.google.com/unimap.edu.my/academicunimap/class-timetable",
    "jadual kuliah": "Jadual kuliah: https://sites.google.com/unimap.edu.my/academicunimap/class-timetable",
    "academic calendar": "Academic Calendar: https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-calendar",
    "kalendar akademik": "Kalendar Akademik: https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-calendar",
    "dress code": "Student dress code: https://www.unimap.edu.my/index.php/en/campus-life/reference/student-dress-code",
    "kod pakaian": "Kod pakaian pelajar: https://www.unimap.edu.my/index.php/en/campus-life/reference/student-dress-code",
    "assistant registrar": "Mdm. Hanimah Karjoo | hanimah@unimap.edu.my | WhatsApp: https://wa.me/601137243477",
    "penolong pendaftar": "Mdm. Hanimah Karjoo | hanimah@unimap.edu.my | WhatsApp: https://wa.me/601137243477",
}

response_cache = {}
CACHE_TTL = 300
conversations = {}

def normalize_query(query):
    return query.lower().strip().replace("?", "").replace("!", "").replace(".", "")

def get_faq_answer(query):
    normalized = normalize_query(query)
    for faq_key, answer in FAQ_CACHE.items():
        if faq_key in normalized or normalized in faq_key:
            return answer
    return None

def retrieve_context(query, top_k=1):
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
    if not chunks:
        return ""
    c = chunks[0]
    return f"📌 {c['source']}: {c['text']}"

def get_cached_response(query):
    key = hashlib.md5(query.lower().encode()).hexdigest()
    cached = response_cache.get(key)
    if cached:
        reply, timestamp = cached
        if time.time() - timestamp < CACHE_TTL:
            return reply
    return None

def save_to_cache(query, reply):
    key = hashlib.md5(query.lower().encode()).hexdigest()
    response_cache[key] = (reply, time.time())

def call_gemini_with_retry(prompt, max_retries=MAX_RETRIES):
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
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            raise

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid request"}), 400

        session_id = data.get("session_id", "default")
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        logger.info(f"[{session_id}] Q: {user_message[:60]}...")

        # Strategy 1: FAQ Cache
        faq_answer = get_faq_answer(user_message)
        if faq_answer:
            return jsonify({"reply": faq_answer, "cached": True})

        # Strategy 2: Response Cache
        cached = get_cached_response(user_message)
        if cached:
            return jsonify({"reply": cached, "cached": True})

        # Strategy 3: RAG Context
        context_chunks = retrieve_context(user_message)
        context_text = format_context(context_chunks)

        if context_text:
            prompt = f"""{SYSTEM_PROMPT}

Retrieved Context:
{context_text}

Student Question: {user_message}

Answer based on the context and system prompt above. Be helpful, concise and accurate."""
        else:
            prompt = f"""{SYSTEM_PROMPT}

Student Question: {user_message}

Answer based on your knowledge above. If unsure, direct to FAQ Centre or relevant contact."""

        # Strategy 4: Call Gemini
        try:
            bot_reply = call_gemini_with_retry(prompt)
            bot_reply = bot_reply.strip()
        except Exception as e:
            logger.warning(f"Gemini failed: {str(e)[:100]}")
            if context_chunks:
                bot_reply = f"📌 {context_chunks[0]['source']}: {context_chunks[0]['text']}\n\nSila rujuk FAQ Centre untuk maklumat lanjut: https://sites.google.com/unimap.edu.my/ur6523003faq/home"
            else:
                bot_reply = "Maaf, sistem sedang sibuk. Sila rujuk FAQ Centre: https://sites.google.com/unimap.edu.my/ur6523003faq/home atau hubungi Mdm. Hanimah: hanimah@unimap.edu.my"

        save_to_cache(user_message, bot_reply)

        if session_id not in conversations:
            conversations[session_id] = []
        conversations[session_id].append({"role": "user", "content": user_message})
        conversations[session_id].append({"role": "assistant", "content": bot_reply})
        conversations[session_id] = conversations[session_id][-MAX_HISTORY:]

        return jsonify({"reply": bot_reply})

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"reply": "Maaf, terdapat masalah teknikal. Sila cuba lagi atau hubungi hanimah@unimap.edu.my"}), 200

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "unimap-fkte-chatbot"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
