import os
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCS_FOLDER = "Docs"


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_documents():
    documents = []

    for filename in os.listdir(DOCS_FOLDER):

        if filename.lower().endswith(".pdf"):

            pdf_path = os.path.join(DOCS_FOLDER, filename)
            reader = PdfReader(pdf_path)

            for page_number, page in enumerate(reader.pages, start=1):

                text = page.extract_text()

                if text:
                    documents.append({
                        "text": clean_text(text),
                        "source": filename,
                        "page": page_number
                    })

    return documents


def retrieve_relevant_chunks(question, documents, top_k=3):

    if not documents:
        return []

    texts = [doc["text"] for doc in documents]

    vectorizer = TfidfVectorizer(stop_words="english")

    vectors = vectorizer.fit_transform(texts + [question])

    document_vectors = vectors[:-1]
    question_vector = vectors[-1]

    similarities = cosine_similarity(
        question_vector,
        document_vectors
    ).flatten()

    ranked_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:

        if similarities[index] > 0:

            result = documents[index].copy()

            result["score"] = float(similarities[index])

            results.append(result)

    return results


def build_context(results):

    context_parts = []

    for result in results:

        context_parts.append(
            f"Source: {result['source']}, Page: {result['page']}\n"
            f"{result['text']}"
        )

    return "\n\n---\n\n".join(context_parts)