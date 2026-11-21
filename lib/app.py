import os
import time
import re

from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from google import genai

from rag import load_documents, retrieve_relevant_chunks, build_context
from prompt import COSINE_PROMPT
from faq import FAQS


load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from the .env file.")

client = genai.Client(api_key=api_key)


# ============================================================
# LOAD UNIVERSITY DOCUMENTS ONCE WHEN SERVER STARTS
# ============================================================

print("Loading university knowledge base...")

DOCUMENTS = load_documents()

print(f"Loaded {len(DOCUMENTS)} PDF pages.")
print(f"Loaded {len(FAQS)} predefined FAQs.")
print("Knowledge base ready.")


# ============================================================
# FAQ NORMALIZATION
# ============================================================

def normalize_question(question):
    """
    Converts a question into a simpler form so that
    small differences in capitalization/punctuation
    do not affect FAQ matching.
    """

    question = question.lower()

    # Remove punctuation
    question = re.sub(r"[^a-z0-9\s]", "", question)

    # Remove extra spaces
    question = re.sub(r"\s+", " ", question).strip()

    return question


# ============================================================
# FAQ MATCHING
# ============================================================

def find_faq(question):
    """
    First tries an exact normalized match.

    Then checks whether the student's question contains
    the predefined FAQ question.

    Returns the FAQ answer if a strong match is found.
    Otherwise returns None.
    """

    normalized_question = normalize_question(question)

    # ----------------------------------------
    # Exact FAQ match
    # ----------------------------------------

    for faq_question, faq_data in FAQS.items():

        normalized_faq = normalize_question(faq_question)

        if normalized_question == normalized_faq:
            return faq_data

    # ----------------------------------------
    # Basic phrase matching
    # ----------------------------------------

    best_match = None
    best_score = 0

    question_words = set(normalized_question.split())

    for faq_question, faq_data in FAQS.items():

        faq_words = set(normalize_question(faq_question).split())

        if not faq_words:
            continue

        common_words = question_words.intersection(faq_words)

        score = len(common_words) / len(faq_words)

        if score > best_score:
            best_score = score
            best_match = faq_data

    # High threshold to avoid giving the wrong FAQ answer
    if best_score >= 0.75:
        return best_match

    return None


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return send_from_directory("../public", "index.html")


# ============================================================
# ASK QUESTION
# ============================================================

@app.route("/ask", methods=["POST"])
def ask():

    total_start = time.time()

    data = request.get_json()

    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "answer": "Please enter a question.",
            "sources": []
        })


    # ========================================================
    # STEP 0: CHECK PREDEFINED FAQS
    # ========================================================

    faq_start = time.time()

    faq_result = find_faq(question)

    faq_time = time.time() - faq_start

    print(f"FAQ matching time: {faq_time:.4f} seconds")


    # ========================================================
    # IF FAQ FOUND → RETURN IMMEDIATELY
    # ========================================================

    if faq_result:

        total_time = time.time() - total_start

        print("FAQ MATCH FOUND")
        print(f"Total response time: {total_time:.4f} seconds")
        print("----------------------------------------")

        return jsonify({
            "question": question,
            "answer": faq_result["answer"],
            "sources": [
                {
                    "document": faq_result["source"],
                    "page": faq_result.get("page", "Not specified")
                }
            ],
            "source_type": "predefined_faq"
        })


    # ========================================================
    # STEP 1: RETRIEVE DOCUMENTS
    # ========================================================

    print("No FAQ match. Starting RAG retrieval...")

    retrieval_start = time.time()

    results = retrieve_relevant_chunks(
        question,
        DOCUMENTS,
        top_k=3
    )

    retrieval_time = time.time() - retrieval_start

    print(f"Retrieval time: {retrieval_time:.2f} seconds")


    # ========================================================
    # IF NOTHING RELEVANT WAS FOUND
    # ========================================================

    if not results:

        total_time = time.time() - total_start

        print(f"Total response time: {total_time:.2f} seconds")
        print("----------------------------------------")

        return jsonify({
            "answer": "I could not verify this from the available official documents.",
            "sources": []
        })


    # ========================================================
    # STEP 2: BUILD CONTEXT
    # ========================================================

    context_start = time.time()

    context = build_context(results)

    context_time = time.time() - context_start

    print(f"Context building time: {context_time:.2f} seconds")


    # ========================================================
    # STEP 3: BUILD PROMPT
    # ========================================================

    prompt = COSINE_PROMPT.format(
        question=question,
        context=context
    )

    print(f"Prompt length: {len(prompt)} characters")


    # ========================================================
    # STEP 4: CALL GEMINI
    # ========================================================

    print("Calling Gemini...")

    gemini_start = time.time()

    try:

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

    except Exception as e:

        print(f"Gemini API error: {e}")

        return jsonify({
            "answer": "The AI service is temporarily unavailable. Please try again in a moment.",
            "sources": []
        }), 503


    gemini_time = time.time() - gemini_start

    print(f"Gemini response time: {gemini_time:.2f} seconds")


    # ========================================================
    # STEP 5: PREPARE SOURCES
    # ========================================================

    sources = [
        {
            "document": result["source"],
            "page": result["page"]
        }
        for result in results
    ]


    # ========================================================
    # STEP 6: TOTAL RESPONSE TIME
    # ========================================================

    total_time = time.time() - total_start

    print(f"Total response time: {total_time:.2f} seconds")
    print("----------------------------------------")


    # ========================================================
    # RETURN RESPONSE
    # ========================================================

    return jsonify({
        "question": question,
        "answer": response.text,
        "sources": sources,
        "source_type": "rag"
    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)