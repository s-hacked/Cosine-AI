from rag import (
    load_documents,
    retrieve_relevant_chunks,
    build_context
)

documents = load_documents()

print(f"\nLoaded {len(documents)} PDF pages.\n")

question = input("Ask a question: ")

results = retrieve_relevant_chunks(
    question,
    documents
)

print("\nRelevant sources:\n")

for result in results:
    print(
        f"- {result['source']} | "
        f"Page {result['page']} | "
        f"Score: {result['score']:.3f}"
    )

print("\nRetrieved context:\n")

print(build_context(results))