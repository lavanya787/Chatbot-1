from nlp_engine.nlp_engine_local import nlp_engine

def test_batch():
    print("📂 Batch NLP Engine Test\n")
    try:
        with open("test_questions.txt", "r") as f:
            questions = f.readlines()
    except FileNotFoundError:
        print("❌ No test_questions.txt file found.")
        return

    for i, q in enumerate(questions, 1):
        question = q.strip()
        if not question:
            continue
        parsed = nlp_engine(question)
        print(f"\n🔹 Question {i}: {question}")
        print("🧠 Structured:")
        for k, v in parsed.items():
            print(f"  {k}: {v}")
        print("-" * 40)

if __name__ == "__main__":
    test_batch()
