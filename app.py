# ------------------- app.py -------------------
import os
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import nltk
from dotenv import load_dotenv
import google.generativeai as genai

# -------- Load .env and configure Gemini --------
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ GOOGLE_API_KEY not found. Create a .env file with GOOGLE_API_KEY=...")

genai.configure(api_key=GEMINI_API_KEY)

# ------------------- NLTK Setup -------------------
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = stopwords.words("english")

# ------------------- Training Data -------------------
data = [
    # Healthy Eating
    ("What are some good high-protein foods?", "eating_tips"),
    ("Healthy breakfast ideas for weight loss", "eating_tips"),
    ("Is intermittent fasting effective?", "eating_tips"),
    ("Tell me about the benefits of a Mediterranean diet", "eating_tips"),
    ("How to reduce sugar intake?", "eating_tips"),
    ("Tips for meal prepping?", "eating_tips"),

    # Fitness & Exercise
    ("What are the best exercises for beginners?", "fitness_tips"),
    ("How to build muscle?", "fitness_tips"),
    ("Best cardio for burning fat?", "fitness_tips"),
    ("What's a good workout routine for a week?", "fitness_tips"),
    ("How to stretch before a workout?", "fitness_tips"),
    ("Advice for staying motivated to exercise", "fitness_tips"),

    # Recipes & Meal Ideas
    ("Can you give me a healthy chicken recipe?", "recipes"),
    ("What's a good salad recipe for lunch?", "recipes"),
    ("Easy and healthy dinner ideas", "recipes"),
    ("How to make a high-protein smoothie?", "recipes"),

    # General Health & Wellness
    ("What are the benefits of drinking water?", "general_health"),
    ("How many hours of sleep do I need?", "general_health"),
    ("Tips for managing stress", "general_health"),
    ("Why is hydration important?", "general_health")
]

train_texts, train_labels = zip(*data)

# ------------------- TF-IDF + Model -------------------
vectorizer = TfidfVectorizer(stop_words=stop_words)
X_train = vectorizer.fit_transform(train_texts)
clf = LogisticRegression()
clf.fit(X_train, train_labels)

# ------------------- Response Handlers -------------------
def handle_eating_tips(_: str) -> str:
    return (
        "**Healthy Eating Tips**\n"
        "- Focus on whole foods: fruits, vegetables, lean proteins, and whole grains.\n"
        "- Stay hydrated throughout the day.\n"
        "- Practice portion control.\n"
        "- Limit processed foods and sugary drinks."
    )

def handle_fitness_tips(_: str) -> str:
    return (
        "**Fitness Tips**\n"
        "- Consistency is key. Start with 2–3 workouts per week.\n"
        "- Mix cardio and strength training.\n"
        "- Always warm up before and cool down after.\n"
        "- Listen to your body—don’t push through sharp pain."
    )

def handle_recipes(_: str) -> str:
    return (
        "**Recipe Ideas**\n"
        "- Quinoa salad with grilled chicken and veggies.\n"
        "- Stir-fry with tofu or lean meat.\n"
        "- High-protein smoothie: Greek yogurt + protein powder + mixed berries."
    )

def handle_general_health(_: str) -> str:
    return (
        "**General Health Info**\n"
        "- Most adults need 7–9 hours of sleep.\n"
        "- Hydration helps energy, brain function, and skin.\n"
        "- Regular exercise and a balanced diet help manage stress."
    )

# ------------------- Gemini API Call -------------------
def get_api_response(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "API key is not configured. Please set GOOGLE_API_KEY in your .env file."
    try:
        # Use a lightweight, fast model. Change if you prefer another.
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            f"You are a helpful health, fitness, and nutrition assistant. Be brief, factual, and friendly.\n\nUser: {prompt}"
        )
        return (response.text or "").strip() or "Sorry, I couldn't generate a response."
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "Sorry, I couldn't fetch the answer. Please try again later."

# ------------------- Flask App -------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"response": "Please type a message."})

    # Predict category
    try:
        X_test = vectorizer.transform([prompt])
        probs = clf.predict_proba(X_test)[0]
        max_prob = float(max(probs))
        category = clf.predict(X_test)[0]
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"response": "I hit a snag classifying that. Try again."})

    CONFIDENCE_THRESHOLD = 0.4  # adjust as you like

    if max_prob > CONFIDENCE_THRESHOLD:
        if category == "eating_tips":
            response_text = handle_eating_tips(prompt)
        elif category == "fitness_tips":
            response_text = handle_fitness_tips(prompt)
        elif category == "recipes":
            response_text = handle_recipes(prompt)
        elif category == "general_health":
            response_text = handle_general_health(prompt)
        else:
            response_text = get_api_response(prompt)
    else:
        response_text = get_api_response(prompt)

    return jsonify({"response": response_text})

# Optional: quick health check
@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "gemini_key_loaded": bool(GEMINI_API_KEY)
    })

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
