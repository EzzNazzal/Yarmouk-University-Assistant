# app.py - Main Flask application file with Telegram Bot Integration

from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os
import sys
import pickle
from pathlib import Path
import requests
import json
import csv
from datetime import datetime
from spellchecker import SpellChecker
from langdetect import detect
import threading
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Conditionally import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    print(f"TensorFlow version: {tf.__version__}")
    TF_AVAILABLE = True
except ImportError:
    print("WARNING: TensorFlow not available. Will run in fallback mode.")
    TF_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    print("WARNING: Deep translator not available. Will run with limited language support.")
    TRANSLATOR_AVAILABLE = False

# Initialize Flask application
app = Flask(__name__)

# Set to track initialization status
app.config['INITIALIZED'] = False

# Telegram Bot Configuration
BOT_TOKEN = '7741799690:AAHvMyriTfBCQjHRj-F1z5_cqxHLZeyNbRQ'

# Download NLTK resources on startup
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

# Initialize the lemmatizer and stopwords
try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
except Exception as e:
    print(f"Warning: Could not initialize NLTK components: {e}")
    lemmatizer = None
    stop_words = set()

# Initialize spell checker
try:
    spell = SpellChecker()
except Exception as e:
    print(f"Warning: Could not initialize spell checker: {e}")
    spell = None

# Set of known names and titles to avoid false corrections
set_of_names = {
    'yarmouk', 'hebah', 'manasrah', 'taani', 'zamil', 'radaideh', 'alquran', 'al', 'raoof', "zu'bi", 'sawsan',
    'abuata', 'el', 'ameera', 'rasool', 'noor', 'yousra', 'ahmad', 'saba', 'alsobeh', 'mohawesh', 'wedyan',
    'abdallah', 'abualbasal', 'aleroud', 'khalid', 'aladeen', 'maryam', 'shakhatreh', 'doush', 'alshawakfa',
    'tariq', 'zeyad', 'barhoush', 'akour', 'shehabat', 'smadi', 'issa', 'daradkeh', 'mustafah', 'nahar',
    'jaafrah', 'rami', 'abrar', 'nahlah', 'nuser', 'magableh', 'zahrawi', 'alguni', 'abed', 'alshboul', 'yanal',
    'hamdan', 'qusay', 'qasem', 'radwan', 'elbashabsheh', 'shannaq', 'sukhni', 'akhras', 'bilal',
    "ra'ed", 'mohammed', 'enas', 'hailat', 'alkhushayni', 'dwairi', 'alsrehin', 'naser', 'ali', 'jaradat',
    'alazzam', 'bsoul', 'faisal', 'hammad', 'malek', 'alhaq', 'klaib', 'alawad', 'sami', 'samarah', 'samer',
    'malkawi', 'hasan', 'mohammad', 'alahmad', 'adnan', 'khatib', 'nawaf', "moy'awiah", 'aldeen', 'maghayreh',
    'wejdan', 'shatnawi', 'abu', 'ottom', 'alikhashashneh', 'rafat', 'iyad', 'amani', 'alkhateeb', 'anas',
    'ashraf', 'eslam', 'alshattnawi', 'yazan', 'yousef', 'alshorman', 'hmoud', 'basima', 'alabed', 'suboh',
    'saifan', 'amal', 'aws', 'rawashdeh', 'emad', 'emran', 'harb', 'shatha',
    'dr.', 'prof.', 'ms.', 'mr.', 'cs', 'da', 'cis', 'bit', 'cys','dr', 'prof', 'ms', 'mr',
}

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Path to log unknown queries
unknown_log_path = os.path.join('data', 'unknown_predictions_log.csv')
if not os.path.exists(unknown_log_path):
    with open(unknown_log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Original Query', 'Processed Query'])

# API Key for OpenRouter
api_key = "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"

class NameMatcher:
    def __init__(self, names_list):
        self.names_list = names_list

    def extract_name_parts(self, query):
        """Extract all name-like parts including possessives and after titles/indicators"""
        titles = ['dr', 'dr.', 'doctor', 'prof', 'prof.', 'professor', 'mr', 'mr.', 'mrs', 'mrs.', 'ms', 'ms.']
        indicators = ['of', 'for', 'about']
        words = query.lower().split()
        name_parts = set()

        for i in range(len(words) - 1):
            if words[i] in titles or words[i] in indicators:
                next_parts = words[i + 1:i + 3]
                for part in next_parts:
                    if part.isalpha():
                        name_parts.add(part)

        for word in words:
            if word.endswith("'s") or word.endswith("'s"):
                base = word[:-2]
                if base.isalpha():
                    name_parts.add(base)

        return list(name_parts)

    def levenshtein_distance(self, str1, str2):
        """Calculate Levenshtein distance between two strings"""
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if str1[i - 1] == str2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[m][n]

    def find_best_match(self, word):
        """Find closest match for a single word"""
        if not word:
            return None

        min_distance = float('inf')
        best_match = None
        for name in self.names_list:
            for part in name.lower().split():
                distance = self.levenshtein_distance(word, part)
                if distance < min_distance:
                    min_distance = distance
                    best_match = part

        threshold = len(word) // 3
        return best_match if min_distance <= threshold else None

    def process(self, query):
        """Process the query and return corrected version"""
        name_parts = self.extract_name_parts(query)
        corrections = {}

        for part in name_parts:
            match = self.find_best_match(part)
            if match:
                corrections[part] = match

        rewritten_query = query.lower()
        for wrong, correct in corrections.items():
            rewritten_query = rewritten_query.replace(wrong, correct)

        return {
            "original_query": query,
            "extracted_names": name_parts,
            "corrections": corrections,
            "rewritten_query": rewritten_query
        }

def correct_spelling(text):
    """Corrects misspellings in text, ignoring known names and titles."""
    words = text.split()
    corrected_words = [
        word if word.lower() in set_of_names or word.istitle()
        else spell.correction(word) or word
        for word in words
    ]
    return ' '.join(corrected_words)

def get_wordnet_pos(tag):
    """Maps POS tag to WordNet POS tag for lemmatization."""
    if tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

def preprocess_question(question):
    """Preprocess a single question: lowercase, tokenize, remove stopwords, lemmatize using POS tagging."""
    preserve_words = {"cs", "ai", "it", "da", "cis", "cys", "bit"}
    question = question.lower()
    question = re.sub(r'\b(\w+)\s*\.\s*(\w+)', r'\1 \2', question)
    question = re.sub(r'(?<=\w)\.(?=\w)', '. ', question)
    tokens = word_tokenize(question)
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    tagged_tokens = nltk.pos_tag(tokens)
    tokens = [
        t[0] if t[0] in preserve_words else lemmatizer.lemmatize(t[0], get_wordnet_pos(t[1]))
        for t in tagged_tokens
    ]
    tokens = [t for t in tokens if t.strip()]
    return tokens

def demonstrate_name_matcher(q):
    q = re.sub(r'(?<=\w)\.(?=\w)', '. ', q)
    q = re.sub(r'(?<=\w)\?', ' ? ', q)
    q = re.sub(r'(?<=\w)\!', ' ! ', q)
    q = re.sub(r'(?<=\w)\-', ' - ', q)
    predefined_names = set_of_names
    matcher = NameMatcher(predefined_names)

    result = matcher.process(q)

    print(f"Original: {result['original_query']}")
    print(f"Extracted: {result['extracted_names']}")
    print(f"Corrections: {result['corrections']}")
    print(f"Rewritten: {result['rewritten_query']}")
    print("-" * 40)

    return result['rewritten_query']

def translate_text(text, src_lang, dest_lang):
    """Translate text from src_lang to dest_lang using GoogleTranslator."""
    if not TRANSLATOR_AVAILABLE:
        return text
        
    try:
        return GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def predict_intent(user_input, original, vectorizer, model, encoder):
    if not vectorizer or not model or not encoder:
        print("Missing required components for prediction")
        return "unknown"
        
    try:
        try:
            input_vector = vectorizer.transform([user_input]).toarray()
        except Exception as ve:
            print(f"Vectorization error: {ve}")
            return "unknown"

        try:
            predicted_probabilities = model.predict(input_vector)[0]
        except Exception as pe:
            print(f"Prediction error: {pe}")
            return "unknown"

        if max(predicted_probabilities) < 0.5:
            unknown_log_path=r"C:\Users\VICTUS\OneDrive\Desktop\chatbot_web\Unkown\unknown_predictions_log.csv"
            with open(unknown_log_path, mode='a', newline='', encoding='utf-8') as log_file:
                writer = csv.writer(log_file)
                writer.writerow([
                    datetime.now().strftime("%d-%m-%Y"),
                    original,
                    user_input
                ])
            return "unknown"

        predicted_class = predicted_probabilities.argmax()
        predicted_intent = encoder.inverse_transform([predicted_class])
        return predicted_intent[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "unknown"

def ask_yu_assistant(user_input, intent, fixed_response=None):
    if intent != 'unknown':
        system_prompt = """
You are a helpful assistant for Yarmouk University.
Your role is to answer only using the information provided.
Do not guess or add anything extra. Be clear, concise, and professional.
- Always reply in the same language used in the user's question.
- Do not apply any formatting such as bold, italic, underlining, bullet points,or any other stylistic elements. Only provide plain text.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
User's question:
{user_input}

Available information:
\"\"\"
{fixed_response}
\"\"\""""
            }
        ]
    else:
        if not app.config['INITIALIZED']:
            return ("I'm currently operating in limited mode due to initialization issues. "
                    "Please contact support or try again later.")
        
        system_prompt = """
You are a helpful assistant for Yarmouk University.
The user is asking a general question. Answer naturally and helpfully.
If the question is not about Yarmouk University, don't answer and say something that you don't know.
- Always reply in the same language used in the user's question.
- Do not apply any formatting such as bold, italic, underlining, bullet points,or any other stylistic elements. Only provide plain text.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": messages
            }),
            timeout=30
        )

        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            reply = re.sub(
                            r'\[.*?\]\((https?://[^\s\)]+)[\.\,\)]*',  # match the URL and optional punctuation
                            r'\n\1',                                   # move URL to a new line
                            reply
                        )

            return reply
        else:
            return f"Error {response.status_code}: Unable to get response. Please try again later."
    except Exception as e:
        return f"Error: {str(e)}. Please try again later."

def get_response(user_input, user_lang, vectorizer, model, encoder, df_responses):
    if not user_input.strip():
        return "It looks like you didn't type anything. How can I help you today?", "unknown"

    if not app.config['INITIALIZED']:
        return ("I'm sorry, but I'm currently operating in limited mode due to initialization issues. "
                "Please contact support or try again later."), "unknown"

    processed_text = demonstrate_name_matcher(user_input)
    processed_text_tokens = preprocess_question(processed_text)
    processed_text = ' '.join(processed_text_tokens)
    if user_lang == 'en' and spell:
        processed_text = correct_spelling(processed_text)
    
    print(processed_text)

    predicted_intent = predict_intent(processed_text, user_input, vectorizer, model, encoder)

    print(predicted_intent)

    if predicted_intent == "unknown" or predicted_intent not in df_responses['Intent'].values:
        return ("I'm not sure I understand your question. Could you please rephrase it "
                "or provide more details?"), "unknown"

    intent_row = df_responses[df_responses['Intent'] == predicted_intent]
    return intent_row.iloc[0]['answer'], predicted_intent

def process_user_input(user_input, vectorizer=None, model=None, encoder=None, df_responses=None):
    """Process user input, detect language, translate if needed, and get response"""
    # Use app config if parameters not provided (for Telegram bot)
    if vectorizer is None:
        vectorizer = app.config.get('vectorizer')
    if model is None:
        model = app.config.get('model')
    if encoder is None:
        encoder = app.config.get('encoder')
    if df_responses is None:
        df_responses = app.config.get('df_responses')
    
    if not user_input.strip():
        return "Please type something to ask."

    if not app.config['INITIALIZED']:
        return "I'm operating in limited mode due to initialization issues. I can still try to help you with general Yarmouk University questions."

    try:
        try:
            user_lang = detect(user_input)
            if user_lang != "en":
                user_lang = 'ar'
        except:
            user_lang = 'en'
            
        if user_lang == "ar" and TRANSLATOR_AVAILABLE:
            translated_input = translate_text(user_input, "ar", "en").lower()
        else:
            translated_input = user_input

        print(f"Translated Input (to English): {translated_input}")
            
        response, intent = get_response(translated_input, user_lang, vectorizer, model, encoder, df_responses)
        
        generated_response = ask_yu_assistant(user_input, intent, response)
        
        return generated_response
    except Exception as e:
        return f"An error occurred: {str(e)}. Please try again."

def initialize_components():
    try:
        vectorizer = None
        model = None
        encoder = None
        df_responses = None
        
        BASE_DIR = Path(__file__).parent
        
        data_dir = BASE_DIR / 'Data'
        model_dir = BASE_DIR / 'Models'
        
        if not data_dir.exists():
            os.makedirs(data_dir)
            print(f"Created missing directory: {data_dir}")
            
        if not model_dir.exists():
            os.makedirs(model_dir)
            print(f"Created missing directory: {model_dir}")
        
        required_files = {
            'FAQs': data_dir / 'FAQs_data.csv',
            'responses': data_dir / 'response_data.csv',
            'model': model_dir / 'FNN_model.keras',
            'encoder': model_dir / 'label_encoder.pkl',
            'vectorizer': model_dir / 'tfidf_vectorizer.pkl'
        }
        
        missing_files = []
        for name, path in required_files.items():
            if not path.exists():
                missing_files.append(str(path))
        
        try:
            df_FAQs = pd.read_csv(required_files['FAQs'])
            df_responses = pd.read_csv(required_files['responses'])
            print("Successfully loaded data files.")
        except Exception as e:
            print(f"ERROR loading data: {e}")
            df_FAQs = pd.DataFrame({'question': ['fallback'], 'intent': ['unknown']})
            df_responses = pd.DataFrame({'Intent': ['unknown'], 'answer': ['I cannot answer that at the moment.']})
        
        try:
            with open(required_files['vectorizer'], 'rb') as vect_file:
                vectorizer = pickle.load(vect_file)
            print(f"Successfully loaded vectorizer from {required_files['vectorizer']}")
            
            with open(required_files['encoder'], 'rb') as enc_file:
                encoder = pickle.load(enc_file)
            print(f"Successfully loaded encoder from {required_files['encoder']}")
        except Exception as e:
            print(f"ERROR loading vectorizer/encoder: {e}")
            if vectorizer is None or encoder is None:
                print("CRITICAL: Failed to load required vectorizer or encoder. The system cannot operate properly.")
            
        if TF_AVAILABLE:
            if required_files['model'].exists():
                try:
                    print("Attempting to load model using standard method...")
                    model = load_model(str(required_files['model']))
                    print("Model loaded successfully!")
                except Exception as e1:
                    print(f"Standard model loading failed: {e1}")
                    
                    try:
                        print("Attempting alternative loading method...")
                        import h5py
                        print(f"Model file exists and is size: {os.path.getsize(required_files['model'])}")
                        with h5py.File(str(required_files['model']), 'r') as f:
                            print(f"H5 file keys: {list(f.keys())}")
                        
                        from tensorflow.keras.models import load_model as keras_load_model
                        model = keras_load_model(str(required_files['model']))
                        print("Model loaded successfully with keras_load_model!")
                    except Exception as e2:
                        print(f"Alternative loading method failed: {e2}")
                        
                        try:
                            print("Attempting to load with compile=False...")
                            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                            model = load_model(str(required_files['model']), compile=False)
                            print("Model loaded successfully with compile=False!")
                        except Exception as e3:
                            print(f"All loading attempts failed: {e3}")
                            print("Will continue without model.")
                            model = None
            else:
                print(f"Model file not found at {required_files['model']}")
                model = None
        else:
            print("TensorFlow not available, will operate in fallback mode")
            model = None
            
        initialization_success = (vectorizer is not None and 
                                 encoder is not None and 
                                 df_responses is not None)
                                 
        return vectorizer, model, encoder, df_responses, initialization_success
        
    except Exception as e:
        print(f"\nINITIALIZATION FAILED: {str(e)}", file=sys.stderr)
        return None, None, None, None, False

# ===================== TELEGRAM BOT HANDLERS =====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hello! I am your Yarmouk Assistant bot. Ask me anything about Yarmouk University.')

async def help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Feel free to ask any questions related to Yarmouk University. I'm here to assist you!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_input = update.message.text.lower()  # Convert to lowercase to handle 'Exit' or 'exit'

    # Show typing indicator while processing
    await update.message.chat.send_action(action="typing")

    # Process the message and get response
    intent = process_user_input(user_input)

    # Send the response
    await update.message.reply_text(intent)

# Main function to run the bot
async def main():
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")

    # Run the bot with polling, which will stop when stop() is called
    await application.run_polling()

# ===================== TELEGRAM BOT SETUP =====================

def run_telegram_bot():
    """Function to run the Telegram bot in a separate thread"""
    print("Starting Telegram bot...")
    
    try:
        import nest_asyncio
        nest_asyncio.apply()
        
        # Run the bot asynchronously
        asyncio.run(main())
        
    except Exception as e:
        print(f"Error starting Telegram bot: {e}")

# ===================== FLASK ROUTES =====================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not app.config['INITIALIZED']:
        response = ask_yu_assistant(user_message, "unknown")
    else:
        response = process_user_input(
            user_message, 
            app.config['vectorizer'], 
            app.config['model'], 
            app.config['encoder'], 
            app.config['df_responses']
        )
    
    return jsonify({'response': response})

@app.route('/status')
def status():
    """Endpoint to check system status"""
    status_info = {
        'initialized': app.config['INITIALIZED'],
        'nltk_available': lemmatizer is not None and len(stop_words) > 0,
        'spell_checker': spell is not None,
        'translator': TRANSLATOR_AVAILABLE,
        'tensorflow': TF_AVAILABLE,
        'model_loaded': app.config['model'] is not None,
        'vectorizer_loaded': app.config['vectorizer'] is not None,
        'encoder_loaded': app.config['encoder'] is not None,
        'telegram_bot': 'Running' if BOT_TOKEN else 'Not configured'
    }
    return jsonify(status_info)

# ===================== MAIN EXECUTION =====================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Yarmouk University Assistant")
    print("=" * 60)
    
    # Initialize components
    print("📊 Initializing ML components...")
    vectorizer, model, encoder, df_responses, init_success = initialize_components()
    
    # Store components in app config
    app.config['vectorizer'] = vectorizer
    app.config['model'] = model
    app.config['encoder'] = encoder
    app.config['df_responses'] = df_responses
    app.config['INITIALIZED'] = init_success
    
    # Report initialization status
    if init_success:
        print("✅ ML components initialized successfully!")
    else:
        print("⚠️ ML components initialized in LIMITED MODE")
        if vectorizer is None:
            print("❌ Failed to load TF-IDF vectorizer")
        if encoder is None:
            print("❌ Failed to load label encoder")
        if model is None:
            print("❌ Failed to load tensorflow model")
        print("The system will use the LLM for basic responses.")
    
    # Start Telegram bot in a separate thread
    if BOT_TOKEN:
        print("🤖 Starting Telegram bot in background...")
        telegram_thread = threading.Thread(target=run_telegram_bot, daemon=True)
        telegram_thread.start()
        print("✅ Telegram bot started!")
    else:
        print("⚠️ Telegram bot token not configured. Skipping Telegram bot.")
    
    # Start Flask application
    print("🌐 Starting Flask web server...")
    print("=" * 60)
    print("🎉 System Ready!")
    print("📱 Telegram Bot: Running" if BOT_TOKEN else "📱 Telegram Bot: Not configured")
    print("🌐 Web Interface: http://localhost:5000")
    print("=" * 60)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        print("👋 Goodbye!")