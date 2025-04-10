import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import spacy
from PyPDF2 import PdfReader
import docx
import nltk
from nltk.tokenize import word_tokenize
import re
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sklearn.preprocessing import StandardScaler
from ml_model.model_runner import predict_from_structured_input as ml_model


# Download NLTK data (run once)
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Store history of models and data
models = {}
data_summaries = {}
raw_data = {}  # Store raw data for line-by-line access

# File parsing functions for different file types
def parse_file(file_path):
    print(f"Attempting to parse file: {file_path}")
    if not os.path.exists(file_path):
        print(f"File not found at: {file_path}")
        return None
    
    file_path = os.path.normpath(file_path)
    file_extension = os.path.splitext(file_path)[1].lower()
    print(f"Detected file extension: {file_extension}")
    data = None
    
    try:
        if file_extension == ".csv":
            data = pd.read_csv(file_path, delimiter=',')
        elif file_extension == ".xlsx":
            data = pd.read_excel(file_path)
        elif file_extension == ".pdf":
            pdf_reader = PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"  # Preserve line breaks
            data = pd.DataFrame({"content": [text]})
        elif file_extension == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])  # Preserve line breaks
            data = pd.DataFrame({"content": [text]})
        elif file_extension in [".txt", ".text"]:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            data = pd.DataFrame({"content": [text]})
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        print(f"Successfully parsed file into DataFrame with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error parsing file: {e}")
        return None

# Preprocess data
def preprocess_data(data):
    if data is not None and not data.empty:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())
        
        text_cols = data.select_dtypes(include=['object']).columns
        for col in text_cols:
            data[col] = data[col].astype(str).apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
        
        return data
    return None

# Train ML model on all rows and columns
def train_combined_model(data, file_id):
    if data is None or data.empty:
        print(f"No data to train model for file {file_id}")
        return None
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    text_cols = data.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) == 0 and (len(text_cols) == 0 or "content" not in data.columns):
        print(f"No numerical or text data found in file {file_id}. Skipping model training.")
        return None
    
    # Dynamically select target as the last numerical column (user can override)
    if len(numeric_cols) > 0:
        target = numeric_cols[-1]
        print(f"Detected potential target column: {target}. Enter a different target column name (or press Enter to use {target}): ")
        user_target = input().strip()
        if user_target and user_target in numeric_cols:
            target = user_target
        elif not user_target:
            print(f"Using {target} as target column.")
        else:
            print(f"Invalid target column. Using {target} as default.")
    else:
        print(f"No numerical columns found. Skipping numerical model training.")
        return None
    
    # Prepare numerical features
    feature_numeric_cols = [col for col in numeric_cols if col != target]
    if not feature_numeric_cols:
        print(f"Insufficient features for training with target {target} in file {file_id}.")
        return None
    
    X_numeric = data[feature_numeric_cols].values
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    
    # Prepare text features line by line
    if len(text_cols) > 0 or "content" in data.columns:
        all_text_cols = text_cols.tolist() if len(text_cols) > 0 else ["content"]
        # Split text into lines and vectorize
        lines = data[all_text_cols].apply(lambda x: x.str.split('\n') if x.name == "content" else x, axis=1)
        flattened_lines = lines.explode(all_text_cols[0]).dropna().tolist()
        tfidf = TfidfVectorizer(max_features=100)
        X_text = tfidf.fit_transform(flattened_lines).toarray()
        # Replicate X_text to match number of rows in data
        if len(X_text) > 0:
            X_text = np.repeat(X_text, len(data) // len(X_text) + 1, axis=0)[:len(data)]
        X = np.hstack((X_numeric_scaled, X_text)) if X_text.size else X_numeric_scaled
    else:
        X = X_numeric_scaled
    
    y = np.log1p(data[target].values)  # Log transform target
    
    # Cross-validation for better accuracy
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LGBMRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
    
    avg_mse = np.mean(mse_scores)
    print(f"Model trained for file {file_id} with average MSE from 5-fold CV: {avg_mse}")
    
    # Final model training on full dataset
    final_model = LGBMRegressor()
    final_model.fit(X, y)
    
    model_path = f"model_{file_id}.joblib"
    joblib.dump(final_model, model_path)
    joblib.dump(scaler, f"scaler_{file_id}.joblib")
    joblib.dump(tfidf, f"tfidf_{file_id}.joblib") if (len(text_cols) > 0 or "content" in data.columns) else None
    
    # Store raw data for later use
    raw_data[file_id] = data
    return final_model, {"features": feature_numeric_cols + all_text_cols, 
                        "target": target, "mse": avg_mse, "model_path": model_path, 
                        "scaler_path": f"scaler_{file_id}.joblib", "tfidf_path": f"tfidf_{file_id}.joblib" if (len(text_cols) > 0 or "content" in data.columns) else None}

# Store trained model and data summary
def store_model_and_summary(model, data, file_id, is_text=False):
    if model and data is not None:
        if is_text:
            model_data, model_path = model
            models[file_id] = {"model_path": model_path, "is_text": is_text}
        else:
            model_data, metadata = model
            model_path = metadata["model_path"]
            models[file_id] = {"model_path": model_path, 
                              "is_text": is_text, 
                              "scaler_path": metadata["scaler_path"], 
                              "tfidf_path": metadata["tfidf_path"]}
        summary = {
            "columns": data.columns.tolist(),
            "shape": data.shape,
            "summary_stats": data.describe(include='all').to_dict()
        }
        data_summaries[file_id] = summary
        summary_path = f"summary_{file_id}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f)
        return f"Model and summary stored for file {file_id} at {model_path} and {summary_path}"
    return None

# Load model
def load_model(file_id):
    if file_id in models:
        model_data = models[file_id]
        if model_data["is_text"]:
            with open(model_data["model_path"], 'r', encoding='utf-8') as f:
                text_data = json.load(f)
            return {"text_content": text_data["text_content"]}
        else:
            model = joblib.load(model_data["model_path"])
            scaler = joblib.load(model_data["scaler_path"])
            tfidf = joblib.load(model_data["tfidf_path"]) if model_data["tfidf_path"] else None
            return {"model": model, "scaler": scaler, "tfidf": tfidf}
    return None

# Parse intent and query
def parse_intent_and_query(user_input):
    doc = nlp(user_input.lower())
    intent = "general"
    query = user_input
    
    if any(token.text in ["hello", "hi", "hey"] for token in doc):
        intent = "greeting"
    elif "upload" in query.lower() and any(token.text in ["train", "upload", "file"] for token in doc):
        intent = "file_processing"
    elif any(token.text in ["what", "average", "sum", "mean"] for token in doc):
        intent = "question"
    print(f"Detected intent: {intent}, Query: {query}")
    return intent, query

# Fetch answer
def fetch_answer(intent, query, data=None):
    print(f"Entering fetch_answer with intent: {intent}, query: {query}")
    if intent == "greeting":
        return "Hello! How can I assist you today?"
    
    elif intent == "question":
        if not models or not data_summaries:
            return "Please upload a file first to train the model."
        
        file_id = max(models.keys()) if models else None
        if file_id is None:
            return "No file has been uploaded yet."
        
        model_data = models[file_id]
        summary = data_summaries[file_id]
        model = load_model(file_id)
        data = raw_data.get(file_id)  # Access raw data
        print(f"Model data: {model_data}, Summary: {summary}")
        
        if model_data["is_text"]:
            text_content = model["text_content"]
            tfidf = TfidfVectorizer().fit_transform([text_content, query])
            similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            if similarity > 0.3:
                doc = nlp(text_content)
                query_doc = nlp(query)
                for token in query_doc:
                    for ent in doc.ents:
                        if token.text.lower() in ent.text.lower():
                            return f"Based on the file, {token.text} relates to: {ent.text}"
                return f"The file contains information similar to your query (similarity: {similarity:.2f})."
            return "No relevant information found in the text."
        
        else:
            all_cols = summary["columns"]
            query_tokens = nlp(query.lower())
            print(f"All Columns: {all_cols}")
            
            aggregate = None
            column_name = None
            for token in query_tokens:
                if token.text in ["sum", "average", "mean"]:
                    aggregate = token.text
                for col in all_cols:
                    if token.text in col.lower().replace('.', '').replace('/', '').replace(' ', ''):
                        column_name = col
                        break
                if column_name:
                    break
            
            if column_name and column_name in data.columns:
                if aggregate == "sum":
                    if pd.api.types.is_numeric_dtype(data[column_name]):
                        return f"Sum of {column_name}: {data[column_name].sum():.2f}"
                    else:
                        return f"Sum of {column_name}: Cannot compute sum on non-numeric data"
                elif aggregate in ["average", "mean"]:
                    if pd.api.types.is_numeric_dtype(data[column_name]):
                        return f"Average of {column_name}: {data[column_name].mean():.2f}"
                    else:
                        return f"Average of {column_name}: Cannot compute average on non-numeric data"
                else:
                    if pd.api.types.is_numeric_dtype(data[column_name]):
                        return f"Summary for {column_name}: mean={data[column_name].mean():.2f}, min={data[column_name].min():.2f}, max={data[column_name].max():.2f}"
                    else:
                        return f"Summary for {column_name}: Non-numeric data"
            else:
                target = summary.get("metadata", {}).get("target")
                numeric_cols = [col for col in all_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
                if target and numeric_cols:
                    input_numeric = np.array([data[col].mean() for col in numeric_cols if col != target]).reshape(1, -1)
                    input_numeric_scaled = model["scaler"].transform(input_numeric)
                    
                    text_cols = [col for col in all_cols if col not in numeric_cols]
                    if model["tfidf"] and len(text_cols) > 0:
                        text_data = ' '.join([data[col].iloc[0] for col in text_cols if not data[col].empty])
                        input_text = model["tfidf"].transform([text_data]).toarray()
                        input_data = np.hstack((input_numeric_scaled, input_text))
                    else:
                        input_data = input_numeric_scaled
                    
                    prediction = model["model"].predict(input_data)
                    prediction = np.expm1(prediction[0])
                    return f"Predicted {target} for your query: {prediction:.2f}"
                return "Please provide a question related to the data columns (e.g., 'what is the closing balance' or 'sum of deposit amt')."

def predict_from_structured_input(structured_input, file_id):
    print(f"🔍 Predicting from NLP-structured input: {structured_input}")
    model_data = models.get(file_id)
    if not model_data:
        return "❌ No trained model found. Please upload and train with a file first."

    model_bundle = load_model(file_id)
    data = raw_data.get(file_id)
    if not data:
        return "❌ No raw data found."

    try:
        # Extract numerical column means
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_inputs = {}
        for feat in structured_input["features"]:
            match = re.findall(r"(\d+\.?\d*)", feat)
            for num in match:
                for col in numeric_cols:
                    if col.lower() in feat.lower():
                        numeric_inputs[col] = float(num)

        input_numeric = np.array([numeric_inputs.get(col, data[col].mean()) for col in numeric_cols]).reshape(1, -1)
        input_numeric_scaled = model_bundle["scaler"].transform(input_numeric)

        text_input = " ".join(structured_input["features"]) + " " + structured_input.get("raw_question", "")
        if model_bundle["tfidf"]:
            input_text = model_bundle["tfidf"].transform([text_input]).toarray()
            final_input = np.hstack((input_numeric_scaled, input_text))
        else:
            final_input = input_numeric_scaled

        prediction = model_bundle["model"].predict(final_input)
        result = np.expm1(prediction[0])
        return f"✅ Predicted {structured_input['target']}: {result:.2f}"

    except Exception as e:
        return f"❌ Error during prediction: {e}"

# Main chatbot loop
def chatbot():
    file_id = 0
    
    while True:
        user_input = input("You: ")
        
        intent, query = parse_intent_and_query(user_input)
        
        if intent == "file_processing" and "upload" in query.lower():
            file_path = input("Please enter the file path: ")
            data = parse_file(file_path)
            if data is not None:
                data = preprocess_data(data)
                print("Data columns and types:\n", data.dtypes)
                if data.select_dtypes(include=[np.number]).empty and ("content" not in data.columns or data["content"].isna().all()):
                    print("No usable numerical or text data found. Skipping training.")
                else:
                    model, metadata = train_combined_model(data, file_id)
                    is_text = False
                    if model:
                        store_model_and_summary((model, metadata), data, file_id, is_text)
                        print(f"File {file_id} processed successfully.")
                        file_id += 1
                    else:
                        print("Failed to train model.")
        
        elif intent == "question":
            answer = fetch_answer(intent, query)
            print(f"Bot: {answer}")

if __name__ == "__main__":
    print("Welcome to the Chatbot! Upload a file or ask a question.")
    chatbot()