from flask import Flask, render_template, request, redirect, session
import sqlite3
import json
import requests
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "ai_interview_secret_key_2026"

# =========================================================
# AI MODEL - SENTENCE TRANSFORMER
# =========================================================
print("----------------------------------------")
print("Loading Sentence Transformer...")
print("----------------------------------------")

ai_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Sentence Transformer Loaded!")
print("----------------------------------------")


# =========================================================
# OLLAMA AI
# =========================================================
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


def ask_ollama(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data.get(
            "response",
            "AI did not return a response."
        )

    except requests.exceptions.ConnectionError:
        return (
            "Ollama is not running. "
            "Please start Ollama using: "
            "ollama run llama3.2:3b"
        )

    except requests.exceptions.Timeout:
        return "AI evaluation timed out. Please try again."

    except Exception as e:
        return f"AI Error: {str(e)}"


# =========================================================
# DATABASE
# =========================================================
def get_db():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            total_score REAL NOT NULL,
            max_score REAL NOT NULL,
            percentage REAL NOT NULL,
            performance TEXT NOT NULL,
            answers_json TEXT NOT NULL,
            strengths_json TEXT NOT NULL,
            weaknesses_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


# =========================================================
# QUESTIONS
# =========================================================
QUESTIONS = [
    {
        "question": "Tell me about yourself.",
        "expected": """
        Candidate should talk about education, technical skills,
        programming skills, projects, interests, achievements,
        career goals and relevant background.
        """
    },
    {
        "question": "Why do you want to work in the field of AI and Machine Learning?",
        "expected": """
        Candidate should explain interest in artificial intelligence,
        machine learning, data, technology, problem solving,
        real world applications and career goals.
        """
    },
    {
        "question": "What are your strengths?",
        "expected": """
        Candidate should describe professional strengths such as
        problem solving, communication, teamwork, leadership,
        adaptability, analytical thinking, discipline and learning
        ability, preferably with examples.
        """
    },
    {
        "question": "What is Python and why is it popular?",
        "expected": """
        Candidate should explain that Python is a high level,
        interpreted programming language. Important concepts include
        simple syntax, readability, libraries, community support,
        automation, web development, data science, artificial
        intelligence and machine learning.
        """
    },
    {
        "question": "Where do you see yourself in five years?",
        "expected": """
        Candidate should explain realistic career goals,
        professional growth, technical expertise, continuous learning,
        leadership, contribution to an organization and long term
        development.
        """
    }
]


# =========================================================
# AI ANSWER EVALUATION
# =========================================================
def evaluate_answer(question_data, answer):

    if not answer or not answer.strip():
        return {
            "score": 0.0,
            "semantic_score": 0.0,
            "coverage_score": 0.0,
            "quality_score": 0.0,
            "depth_score": 0.0,
            "specificity_score": 0.0,
            "feedback": "No answer was provided. Please answer the question.",
            "ai_feedback": "No answer was provided.",
            "concepts_found": [],
            "missing_concepts": [],
            "strengths": [],
            "improvements": []
        }

    answer = str(answer).strip()
    expected_text = str(question_data.get("expected", ""))
    question = str(question_data.get("question", "")).lower()
    answer_lower = answer.lower()

    # =====================================================
    # SENTENCE TRANSFORMER SEMANTIC SCORE
    # =====================================================

    answer_embedding = ai_model.encode(
        [answer],
        normalize_embeddings=True
    )

    expected_embedding = ai_model.encode(
        [expected_text],
        normalize_embeddings=True
    )

    similarity = float(
        cosine_similarity(
            answer_embedding,
            expected_embedding
        )[0][0]
    )

    semantic_score = float(
        round(
            max(0.0, min(10.0, similarity * 10.0)),
            2
        )
    )

    # =====================================================
    # CONCEPTS
    # =====================================================

    if "python" in question:

        concepts = [
            "programming",
            "language",
            "syntax",
            "readability",
            "library",
            "community",
            "data",
            "machine learning",
            "artificial intelligence",
            "automation"
        ]

    elif "machine learning" in question:

        concepts = [
            "artificial intelligence",
            "machine learning",
            "data",
            "model",
            "prediction",
            "problem solving",
            "technology",
            "career"
        ]

    elif "strength" in question:

        concepts = [
            "problem solving",
            "communication",
            "teamwork",
            "leadership",
            "learning",
            "adaptability",
            "discipline",
            "analytical"
        ]

    elif "five years" in question:

        concepts = [
            "career",
            "growth",
            "skills",
            "learning",
            "leadership",
            "experience",
            "goals",
            "contribution"
        ]

    else:

        concepts = [
            "education",
            "skills",
            "project",
            "experience",
            "interest",
            "career",
            "goal"
        ]

    found_concepts = [
        concept
        for concept in concepts
        if concept.lower() in answer_lower
    ]

    coverage_ratio = (
        len(found_concepts) / len(concepts)
        if concepts
        else 0.0
    )

    coverage_score = float(
        round(
            coverage_ratio * 10.0,
            2
        )
    )

    # =====================================================
    # DEPTH
    # =====================================================

    words = answer.split()
    word_count = len(words)

    sentences = [
        s.strip()
        for s in answer
        .replace("!", ".")
        .replace("?", ".")
        .split(".")
        if s.strip()
    ]

    sentence_count = len(sentences)

    if word_count < 10:
        depth_score = 2.0

    elif word_count < 20:
        depth_score = 4.0

    elif word_count < 35:
        depth_score = 6.0

    elif word_count < 60:
        depth_score = 8.0

    else:
        depth_score = 10.0

    # =====================================================
    # SPECIFICITY
    # =====================================================

    example_words = [
        "for example",
        "example",
        "such as",
        "because",
        "therefore",
        "which",
        "when",
        "where",
        "using",
        "used",
        "built",
        "developed",
        "worked",
        "project"
    ]

    specificity_hits = sum(
        1
        for word in example_words
        if word in answer_lower
    )

    if specificity_hits >= 4:
        specificity_score = 10.0

    elif specificity_hits == 3:
        specificity_score = 8.0

    elif specificity_hits == 2:
        specificity_score = 6.0

    elif specificity_hits == 1:
        specificity_score = 4.0

    else:
        specificity_score = 2.0

    # =====================================================
    # QUALITY
    # =====================================================

    quality_score = float(
        round(
            (
                depth_score +
                specificity_score +
                coverage_score
            ) / 3,
            2
        )
    )

    # =====================================================
    # FINAL SCORE
    # =====================================================

    final_score = float(
        round(
            max(
                0.0,
                min(
                    10.0,
                    semantic_score * 0.45 +
                    coverage_score * 0.25 +
                    depth_score * 0.15 +
                    specificity_score * 0.15
                )
            ),
            1
        )
    )

    # =====================================================
    # STRENGTHS
    # =====================================================

    strengths = []

    if semantic_score >= 8:
        strengths.append("Highly relevant answer")

    elif semantic_score >= 6:
        strengths.append("Good relevance to the question")

    if coverage_score >= 7:
        strengths.append("Good concept coverage")

    if depth_score >= 8:
        strengths.append("Good answer depth")

    if specificity_score >= 8:
        strengths.append(
            "Good use of specific explanations or examples"
        )

    if sentence_count >= 3:
        strengths.append("Well-developed response")

    if not strengths:
        strengths.append(
            "The answer provides a basic response"
        )

    # =====================================================
    # MISSING CONCEPTS
    # =====================================================

    missing_concepts = [
        concept
        for concept in concepts
        if concept.lower() not in answer_lower
    ]

    # =====================================================
    # IMPROVEMENTS
    # =====================================================

    improvements = []

    if semantic_score < 6:
        improvements.append(
            "Make your answer more directly related to the question."
        )

    if coverage_score < 6:
        improvements.append(
            "Include more important concepts related to the topic."
        )

    if depth_score < 6:
        improvements.append(
            "Explain your answer in more detail."
        )

    if specificity_score < 6:
        improvements.append(
            "Add a specific example or practical explanation."
        )

    if sentence_count <= 1:
        improvements.append(
            "Use multiple sentences to explain your thoughts clearly."
        )

    if not improvements:
        improvements.append(
            "Keep your answer structured and continue adding relevant examples."
        )

    # =====================================================
    # BASIC FEEDBACK
    # =====================================================

    feedback_parts = []

    if semantic_score >= 8:

        feedback_parts.append(
            "Your answer is highly relevant to the question."
        )

    elif semantic_score >= 6:

        feedback_parts.append(
            "Your answer is relevant, but it can be explained more clearly."
        )

    else:

        feedback_parts.append(
            "Your answer needs stronger relevance to the question."
        )

    if coverage_score >= 7:

        feedback_parts.append(
            "You covered most of the important concepts."
        )

    elif coverage_score >= 4:

        feedback_parts.append(
            "You covered some important concepts."
        )

    else:

        feedback_parts.append(
            "Several important concepts are missing."
        )

    if depth_score >= 8:

        feedback_parts.append(
            "The answer has good depth."
        )

    elif depth_score >= 5:

        feedback_parts.append(
            "The answer has moderate depth."
        )

    else:

        feedback_parts.append(
            "The answer needs more explanation."
        )

    if specificity_score >= 8:

        feedback_parts.append(
            "Good use of specific details or examples."
        )

    else:

        feedback_parts.append(
            "Adding examples or specific details would make the answer stronger."
        )

    if missing_concepts:

        feedback_parts.append(
            "Important concepts you could include: "
            + ", ".join(missing_concepts[:4])
            + "."
        )

    basic_feedback = " ".join(feedback_parts)

    # =====================================================
    # OLLAMA AI EVALUATION
    # =====================================================
    # =====================================================
    # PERSONALIZED OLLAMA AI EVALUATION
    # =====================================================

    found_text = (
        ", ".join(found_concepts)
        if found_concepts
        else "None"
    )

    missing_text = (
        ", ".join(missing_concepts[:8])
        if missing_concepts
        else "None"
    )

    ollama_prompt = f"""
You are an expert interview evaluator.

Your job is to evaluate ONLY the candidate's actual answer.
Your feedback MUST be specific to this exact question and this exact answer.

INTERVIEW QUESTION:
{question_data["question"]}

EXPECTED ANSWER AREAS:
{expected_text}

CANDIDATE'S ACTUAL ANSWER:
{answer}

AUTOMATED SCORE:
{final_score}/10

SEMANTIC SCORE:
{semantic_score}/10

CONCEPT COVERAGE:
{coverage_score}/10

ANSWER DEPTH:
{depth_score}/10

SPECIFICITY:
{specificity_score}/10

CONCEPTS FOUND IN THE ANSWER:
{found_text}

CONCEPTS MISSING FROM THE ANSWER:
{missing_text}

IMPORTANT RULES:

1. Do NOT give generic or repeated feedback.
2. Read the candidate's actual answer carefully.
3. Mention specific things the candidate actually said.
4. Identify what was correct in THIS answer.
5. Identify what is missing from THIS answer.
6. If the answer is incorrect, clearly explain the incorrect part.
7. If the answer is very short, explain exactly what should be added.
8. If the answer contains a good example, mention that example.
9. If the answer contains no example, say that an example would improve it.
10. Do NOT copy the same feedback structure or sentences for every question.
11. Do NOT invent information that the candidate did not say.
12. Do NOT change the automated numerical score.
13. Do NOT judge grammar harshly unless it affects understanding.
14. Keep the feedback useful for an actual interview.

Return exactly these sections:

AI FEEDBACK:
Give 2-4 sentences explaining how THIS specific answer performed.

WHAT YOU DID WELL:
Give 1-3 specific points based only on the candidate's answer.

WHAT IS MISSING:
Mention the most important missing concepts or details for THIS question.

HOW TO IMPROVE:
Give specific advice for improving THIS exact answer.

BETTER ANSWER:
Write a concise example of how the candidate could answer THIS question better.

Remember:
The feedback must depend on the candidate's actual answer.
Different answers must produce different feedback.
Do not use generic praise such as "Good answer" without explaining why.
"""

    ai_feedback = ask_ollama(ollama_prompt)

    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {
        "score": float(final_score),
        "semantic_score": float(semantic_score),
        "coverage_score": float(coverage_score),
        "quality_score": float(quality_score),
        "depth_score": float(depth_score),
        "specificity_score": float(specificity_score),
        "feedback": str(basic_feedback),
        "ai_feedback": str(ai_feedback),
        "concepts_found": list(found_concepts),
        "missing_concepts": list(missing_concepts),
        "strengths": list(strengths),
        "improvements": list(improvements)
    }


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    if "user_id" in session:
        return redirect("/dashboard")

    return redirect("/login")


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if not name or not email or not password:

            return "Please fill all fields."

        hashed_password = generate_password_hash(
            password
        )

        conn = get_db()

        try:

            conn.execute(
                """
                INSERT INTO users (
                    name,
                    email,
                    password
                )
                VALUES (?, ?, ?)
                """,
                (
                    name,
                    email,
                    hashed_password
                )
            )

            conn.commit()

        except sqlite3.IntegrityError:

            conn.close()

            return "Email already registered. Please login."

        conn.close()

        return redirect("/login")

    return render_template("register.html")


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        conn = get_db()

        user = conn.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        conn.close()

        if user and check_password_hash(
            user["password"],
            password
        ):

            session["user_id"] = user["id"]
            session["name"] = user["name"]
            session["email"] = user["email"]

            session.pop(
                "current_question",
                None
            )

            session.pop(
                "answers",
                None
            )

            session.pop(
                "scores",
                None
            )

            return redirect("/dashboard")

        return "Invalid email or password."

    return render_template("login.html")


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect("/login")

    return render_template(
        "dashboard.html",
        name=session.get("name")
    )


# =========================================================
# PERFORMANCE
# =========================================================

@app.route("/performance")
def performance():

    if "user_id" not in session:
        return redirect("/login")

    conn = get_db()

    interviews_rows = conn.execute(
        """
        SELECT *
        FROM interviews
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (session["user_id"],)
    ).fetchall()

    conn.close()

    interviews = []

    for row in interviews_rows:

        item = dict(row)

        item["answers"] = json.loads(
            item["answers_json"]
        )

        item["strengths"] = json.loads(
            item["strengths_json"]
        )

        item["weaknesses"] = json.loads(
            item["weaknesses_json"]
        )

        interviews.append(item)

    average_percentage = 0.0
    best_percentage = 0.0
    improvement = 0.0

    if interviews:

        average_percentage = round(
            sum(
                float(x["percentage"])
                for x in interviews
            ) / len(interviews),
            1
        )

        best_percentage = round(
            max(
                float(x["percentage"])
                for x in interviews
            ),
            1
        )

    if len(interviews) >= 2:

        improvement = round(
            float(interviews[0]["percentage"])
            -
            float(interviews[1]["percentage"]),
            1
        )

    return render_template(
        "performance.html",
        interviews=interviews,
        average_percentage=average_percentage,
        best_percentage=best_percentage,
        improvement=improvement,
        name=session.get("name")
    )


# =========================================================
# INTERVIEW
# =========================================================

@app.route("/interview")
def interview():

    if "user_id" not in session:
        return redirect("/login")

    if "current_question" not in session:

        session["current_question"] = 0
        session["answers"] = []
        session["scores"] = []

    current_question = int(
        session["current_question"]
    )

    if current_question >= len(QUESTIONS):

        session.pop(
            "current_question",
            None
        )

        session.pop(
            "answers",
            None
        )

        session.pop(
            "scores",
            None
        )

        return redirect("/interview")

    question_data = QUESTIONS[
        current_question
    ]

    return render_template(
        "interview.html",
        name=session.get("name"),
        question=question_data["question"],
        question_number=current_question + 1,
        total_questions=len(QUESTIONS)
    )


# =========================================================
# EVALUATE ANSWER
# =========================================================

@app.route("/evaluate", methods=["POST"])
def evaluate():

    if "user_id" not in session:
        return redirect("/login")

    question = request.form.get(
        "question",
        ""
    ).strip()

    answer = request.form.get(
        "answer",
        ""
    ).strip()

    current_question = int(
        session.get(
            "current_question",
            0
        )
    )

    if current_question >= len(QUESTIONS):

        session.pop(
            "current_question",
            None
        )

        return redirect("/interview")

    question_data = QUESTIONS[
        current_question
    ]

    evaluation = evaluate_answer(
        question_data,
        answer
    )

    score = float(
        evaluation["score"]
    )

    semantic_score = float(
        evaluation["semantic_score"]
    )

    coverage_score = float(
        evaluation["coverage_score"]
    )

    quality_score = float(
        evaluation["quality_score"]
    )

    feedback = str(
        evaluation["feedback"]
    )

    ai_feedback = str(
        evaluation["ai_feedback"]
    )

    answers = session.get(
        "answers",
        []
    )

    scores = session.get(
        "scores",
        []
    )

    answers.append({

        "question": str(question),

        "answer": str(answer),

        "score": float(score),

        "feedback": feedback,

        "ai_feedback": ai_feedback,

        "semantic_score": float(
            semantic_score
        ),

        "coverage_score": float(
            coverage_score
        ),

        "quality_score": float(
            quality_score
        ),

        "depth_score": float(
            evaluation["depth_score"]
        ),

        "specificity_score": float(
            evaluation["specificity_score"]
        ),

        "strengths": list(
            evaluation["strengths"]
        ),

        "improvements": list(
            evaluation["improvements"]
        ),

        "missing_concepts": list(
            evaluation["missing_concepts"]
        )
    })

    scores.append(
        float(score)
    )

    session["answers"] = answers
    session["scores"] = scores

    # =====================================================
    # NEXT QUESTION
    # =====================================================

    if current_question < len(QUESTIONS) - 1:

        session["current_question"] = (
            current_question + 1
        )

        return redirect("/interview")

    # =====================================================
    # FINAL RESULT
    # =====================================================

    total_score = float(
        round(
            sum(scores),
            1
        )
    )

    max_score = float(
        len(QUESTIONS) * 10
    )

    percentage = float(
        round(
            (
                total_score /
                max_score
            ) * 100,
            1
        )
    )

    if percentage >= 85:

        performance_text = "Excellent"

    elif percentage >= 70:

        performance_text = "Very Good"

    elif percentage >= 55:

        performance_text = "Good"

    elif percentage >= 40:

        performance_text = "Needs Improvement"

    else:

        performance_text = (
            "Needs Significant Improvement"
        )

    strengths = [

        item["question"]

        for item in answers

        if float(item["score"]) >= 8

    ]

    weaknesses = [

        item["question"]

        for item in answers

        if float(item["score"]) < 6

    ]

    # =====================================================
    # SAVE INTERVIEW
    # =====================================================

    conn = get_db()

    conn.execute(
        """
        INSERT INTO interviews (
            user_id,
            total_score,
            max_score,
            percentage,
            performance,
            answers_json,
            strengths_json,
            weaknesses_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session["user_id"],
            float(total_score),
            float(max_score),
            float(percentage),
            performance_text,
            json.dumps(answers),
            json.dumps(strengths),
            json.dumps(weaknesses),
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    conn.commit()
    conn.close()

    result_answers = list(
        answers
    )

    # Clear current interview

    session.pop(
        "current_question",
        None
    )

    session.pop(
        "answers",
        None
    )

    session.pop(
        "scores",
        None
    )

    return render_template(
        "result.html",
        name=session.get("name"),
        answers=result_answers,
        total_score=total_score,
        max_score=max_score,
        percentage=percentage,
        performance=performance_text,
        strengths=strengths,
        weaknesses=weaknesses
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    init_db()

    print("----------------------------------------")
    print("ADVANCED AI INTERVIEW SYSTEM")
    print("----------------------------------------")
    print("Sentence Transformer: ENABLED")
    print("Ollama AI: ENABLED")
    print("Ollama Model:", OLLAMA_MODEL)
    print("Evaluation: Semantic + Concepts + Quality + Ollama")
    print("Questions:", len(QUESTIONS))
    print("----------------------------------------")
    print("Server running at:")
    print("http://127.0.0.1:5000")
    print("----------------------------------------")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )