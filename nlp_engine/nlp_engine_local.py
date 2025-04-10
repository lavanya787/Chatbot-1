import nltk
import spacy
from transformers import pipeline

# Downloads
nltk.download('punkt')
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Zero-shot classifier
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Intent list
INTENTS = [
    "predict", "classify", "analyze", "summarize", "compare", "explain",
    "retrieve", "filter", "visualize", "evaluate", "optimize", "debug", "correlate", "validate"
]

# Keywords
KEY_TARGET_WORDS = [
    "price", "score", "result", "grade", "output", "target", "label", "prediction",
    "salary", "value", "accuracy", "performance", "trend", "relation"
]
OUTPUT_FORMATS = ["value", "label", "probability", "number", "range", "percentage"]

# Greeting check
def is_greeting(text):
    return text.lower().strip() in ["hi", "hello", "hey", "thanks", "thank you", "ok", "okay"]

# Intent classifier
def extract_intent(question):
    try:
        result = zero_shot_classifier(question, INTENTS)
        return result['labels'][0]
    except:
        return "analyze"

# Feature extractor
def extract_features_and_target(text):
    doc = nlp(text)
    features, target = [], None
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        if any(word in chunk_text for word in KEY_TARGET_WORDS):
            target = chunk.text
        else:
            features.append(chunk.text)
    return list(set(features)), target

# Condition extractor
def extract_conditions(text):
    keywords = ['if', 'where', 'when', 'greater', 'less', 'more than', 'equal', 'above', 'below', 'between']
    tokens = nltk.word_tokenize(text.lower())
    return [w for w in tokens if w in keywords] or None

# Output format
def extract_output_format(text):
    for fmt in OUTPUT_FORMATS:
        if fmt in text.lower():
            return fmt
    return "label"

# Full NLP Engine
def nlp_engine(question: str) -> dict:
    question = question.strip()
    if is_greeting(question):
        return {
            "intent": "none",
            "features": [],
            "target": None,
            "conditions": None,
            "output_format": None,
            "raw_question": question
        }

    intent = extract_intent(question)
    features, target = extract_features_and_target(question)
    conditions = extract_conditions(question)
    output_format = extract_output_format(question)

    # Fallback values if missing
    if not features:
        features = ["unknown feature"]
    if not target:
        target = "unknown target"

    return {
        "intent": intent,
        "features": features,
        "target": target,
        "conditions": conditions,
        "output_format": output_format,
        "raw_question": question
    }

# CLI
if __name__ == "__main__":
    print("NLP Engine - Ask a question to analyze:")
    q = input("👉 ")
    output = nlp_engine(q)
    print("\n Structured input to send to model:")
    for k, v in output.items():
        print(f"{k}: {v}")
