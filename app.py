import os
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__)

# OpenRouter API key
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY","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")
)

SYSTEM_PROMPT = """
You are a helpful university assistant chatbot for UniMAP (Universiti Malaysia Perlis).
You specifically help students from the Faculty of Electrical Engineering & Technology (FKTE),
Bachelor of Mechatronic Engineering with Honours programme (UR6523003).

You help students with questions about:
1. COURSE REGISTRATION - How to register, add or drop subjects
2. ACADEMIC FORMS - Where to find and how to fill forms
3. CURRICULUM STRUCTURE - Course requirements and structure
4. ACADEMIC RULES - Regulations and policies
5. DEADLINES - Important academic dates
6. CONTACTS - Who to refer to for specific issues

IMPORTANT UNIMAP SPECIFIC INFO:
- Student portal is called i-Ma'luum at https://imaluum.unimap.edu.my
- Students log in using their Matrix Number and password
- To drop a subject go to i-Ma'luum → Academic → Course Registration
- For academic problems contact FKTE at ftke@unimap.edu.my
- UniMAP main campus is in Arau, Perlis

IMPORTANT REFERENCE LINKS (share these when relevant):
- UniMAP Website: https://www.unimap.edu.my/
- FKTE Website: https://ftke.unimap.edu.my/
- Academic Calendar: https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-calendar
- Academic Guidebook: https://www.unimap.edu.my/index.php/en/campus-life/reference/academic-guide-book
- Class Timetable: https://sites.google.com/unimap.edu.my/academicunimap/class-timetable
- FAQ Centre: https://sites.google.com/unimap.edu.my/ur6523003faq/home

GUIDELINES:
- Always be friendly, polite and helpful
- Answer in the same language the student uses (Malay or English)
- If you don't know the exact answer, refer them to the FAQ Centre
- Always provide relevant links when answering
"""

# Store conversation history
conversations = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id", "default")
    user_message = data.get("message", "")

    # Initialize conversation if new session
    if session_id not in conversations:
        conversations[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    # Add user message
    conversations[session_id].append({
        "role": "user",
        "content": user_message
    })

    # Get response from AI
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=conversations[session_id]
    )

    bot_reply = response.choices[0].message.content

    # Add bot reply to history
    conversations[session_id].append({
        "role": "assistant",
        "content": bot_reply
    })

    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)