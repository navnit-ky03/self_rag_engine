
from retriever import retriver

questions = [
    "Tell me about Navnit's technical skills.",
    "What projects has Navnit worked on?",
    "Which technologies were used in his projects?",
]

for question in questions:
    print(f"\n{'=' * 60}")
    print(f"QUESTION: {question}")

    try:
        docs = retriver.invoke(question)

        if not docs:
            print("No documents retrieved.")
            continue

        for i, doc in enumerate(docs, start=1):
            print(f"\n--- Result {i} ---")
            print(doc.page_content[:800])
            print("Metadata:", doc.metadata)

    except Exception as exc:
        print(f"Retrieval failed: {exc}")
