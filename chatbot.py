import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import pickle
import os
from pathlib import Path
import PyPDF2  # For PDF file reading
import csv  # For CSV file reading
import json  # For JSON file reading
try:
    import pandas as pd  # For CSV, JSON, and XLSX (install with: pip install pandas openpyxl)
except ImportError:
    pd = None
try:
    import docx  # For DOCX files (install with: pip install python-docx)
except ImportError:
    docx = None

# Load spaCy NLP model (small English model)
nlp = spacy.load("en_core_web_sm")

# Define a class for the Chatbot ML Model
class ChatbotModel:
    def __init__(self, model_dir="trained_models"):
        """
        Initialize the chatbot model.
        model_dir: Directory to save the trained model
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model_file = self.model_dir / "trained_model.pkl"
        
        self.vectorizer = TfidfVectorizer()
        self.classifier = SGDClassifier(loss="log_loss", warm_start=True)
        self.pipeline = None
        self.is_trained = False
        
        # Reset existing model to start fresh
        if self.model_file.exists():
            os.remove(self.model_file)
            print("Removed existing model to start fresh.")
        self._load_model()

    def _load_model(self):
        """Load the trained model from disk if it exists."""
        if self.model_file.exists():
            with open(self.model_file, "rb") as f:
                self.pipeline = pickle.load(f)
            self.is_trained = True
            print("Loaded existing model from disk.")
        else:
            self.pipeline = Pipeline([
                ("tfidf", self.vectorizer),
                ("clf", self.classifier)
            ])

    def _save_model(self):
        """Save the trained model to disk."""
        with open(self.model_file, "wb") as f:
            pickle.dump(self.pipeline, f)
        print("Model saved to disk.")

    def preprocess_text(self, text):
        """Preprocess text using spaCy for feature extraction."""
        doc = nlp(text.lower())
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(tokens)

    def process_nlp_response(self, nlp_response):
        """Extract features and intent from NLP engine response."""
        features = self.preprocess_text(nlp_response.get("features", ""))
        intent = nlp_response.get("intent", "")
        return features, intent

    def train_from_input(self, nlp_response):
        """Train the model incrementally with NLP engine response."""
        features, intent = self.process_nlp_response(nlp_response)
        
        if not self.is_trained:
            if not hasattr(self, "initial_data"):
                self.initial_data = {"X": [], "y": []}
            
            self.initial_data["X"].append(features)
            self.initial_data["y"].append(intent)
            
            if len(set(self.initial_data["y"])) > 1:
                self.pipeline.fit(self.initial_data["X"], self.initial_data["y"])
                self.is_trained = True
                del self.initial_data
                self._save_model()
            else:
                print("Need at least 2 different intents to start training. Collecting data...")
        else:
            X = [features]
            y = [intent]
            classes = np.unique(list(self.pipeline.classes_) + [intent])
            self.pipeline.named_steps["clf"].partial_fit(
                self.pipeline.named_steps["tfidf"].transform(X), y, classes=classes
            )
            self._save_model()

    def predict_intent(self, user_input):
        """Predict the intent of the user input using NLP-processed features."""
        nlp_response = {
            "intent": "",  # To be filled by NLP engine
            "features": user_input,
            "target": "",
            "conditions": "",
            "output_format": "",
            "raw_question": user_input
        }
        features, _ = self.process_nlp_response(nlp_response)
        if not self.is_trained:
            return None
        prediction = self.pipeline.predict([features])[0]
        confidence = self.pipeline.predict_proba([features]).max()
        return prediction, confidence

    def process_uploaded_file(self, file_path):
        """Process user-uploaded files and extract text, then simulate NLP response."""
        file_ext = Path(file_path).suffix.lower()
        text_content = ""

        try:
            if file_ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            elif file_ext == ".pdf":
                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() or ""
            elif file_ext == ".csv":
                if pd is not None:
                    df = pd.read_csv(file_path)
                    text_content = " ".join(df.astype(str).values.flatten())
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        csv_reader = csv.reader(f)
                        for row in csv_reader:
                            text_content += " ".join(row) + " "
            elif file_ext == ".json":
                if pd is not None:
                    df = pd.read_json(file_path)
                    text_content = " ".join(df.astype(str).values.flatten())
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        text_content = " ".join(str(data).split())
            elif file_ext == ".docx" and docx is not None:
                doc = docx.Document(file_path)
                text_content = " ".join([paragraph.text for paragraph in doc.paragraphs])
            elif file_ext == ".xlsx" and pd is not None:
                df = pd.read_excel(file_path)  # Requires openpyxl or xlrd
                text_content = " ".join(df.astype(str).values.flatten())
            else:
                print(f"Unsupported or missing library for file type: {file_ext}. "
                      "Install required libraries (e.g., pip install pandas openpyxl) for full support.")
                return

            nlp_response = {
                "intent": Path(file_path).stem,  # Use file name as intent
                "features": text_content,
                "target": "",  # Placeholder
                "conditions": "",  # Placeholder
                "output_format": "",  # Placeholder
                "raw_question": text_content  # Use file content as raw question
            }
            processed_features, intent = self.process_nlp_response(nlp_response)
            self.train_from_input(nlp_response)
            print(f"Processed and trained on file: {file_path} with intent '{intent}'")

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    def handle_query(self, user_input, file_paths=None):
        """Handle user query and return response, optionally processing multiple files."""
        if file_paths:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    self.process_uploaded_file(file_path)
        
        result = self.predict_intent(user_input)
        if result is None or result[1] < 0.7:
            return "I don’t know the answer yet. Please provide more input or a file to learn from."
        else:
            return f"Predicted intent: {result[0]} (Confidence: {result[1]:.2f})"

# Interactive Testing
if __name__ == "__main__":
    chatbot = ChatbotModel()
    
    print("Chatbot Testing Mode. Commands:")
    print("- Type a query to test prediction (e.g., 'What’s the weather?')")
    print("- To train, type 'train:<input>:<intent>' (e.g., 'train:How are you?:greeting')")
    print("- To test with files, type 'file:<query>:<file_path1>:<file_path2>:...' (e.g., 'file:What’s in it?:path1:path2')")
    print("- Type 'exit' to quit")

    while True:
        user_input = input("\nEnter command: ").strip()
        
        if user_input.lower() == "exit":
            break
        
        # Training command
        if user_input.startswith("train:"):
            try:
                _, text, intent = user_input.split(":", 2)
                nlp_response = {
                    "intent": intent,
                    "features": text,
                    "target": "",
                    "conditions": "",
                    "output_format": "",
                    "raw_question": text
                }
                chatbot.train_from_input(nlp_response)
                print(f"Trained on: '{text}' with intent '{intent}'")
            except ValueError:
                print("Invalid format. Use 'train:<input>:<intent>'")
        
        # File command
        elif user_input.startswith("file:"):
            try:
                parts = user_input.split(":", 2)
                if len(parts) < 3:
                    raise ValueError
                _, query, file_paths_str = parts
                file_paths = file_paths_str.split(":")
                response = chatbot.handle_query(query, file_paths=file_paths)
                print(response)
            except ValueError:
                print("Invalid format. Use 'file:<query>:<file_path1>:<file_path2>:...'")
        
        # Prediction command
        else:
            response = chatbot.handle_query(user_input)
            print(response)