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
    ("How many hours of sleep do I need?", "general_health"),
    ("Tips for managing stress", "general_health"),
    ("Why is hydration important?", "general_health"),

    # Water Benefits
    ("What are the benefits of drinking water?", "water_benefits"),
    ("Why should I drink water?", "water_benefits"),
    ("Is drinking water good for health?", "water_benefits")
]

train_texts, train_labels = zip(*data)

# ------------------- TF-IDF + Model -------------------
vectorizer = TfidfVectorizer(stop_words=stop_words)
X_train = vectorizer.fit_transform(train_texts)
clf = LogisticRegression()
clf.fit(X_train, train_labels)

# ------------------- Response Handlers with QA dictionaries -------------------
def handle_eating_tips(question: str) -> str:
    qa = {
        "what are some good high-protein foods": "- Include eggs, chicken, fish, lentils, beans, Greek yogurt, and tofu.",
        "healthy breakfast ideas for weight loss": "- Try oatmeal with fruits, smoothies, or egg whites with veggies.",
        "is intermittent fasting effective": "- It can help with weight control if done safely and consistently.",
        "tell me about the benefits of a mediterranean diet": "- Rich in fruits, vegetables, healthy fats, and reduces heart disease risk.",
        "how to reduce sugar intake": "- Avoid sugary drinks, read labels, and choose natural sweeteners.",
        "tips for meal prepping": "- Plan meals in advance, use containers, and cook in batches."
    }
    key = question.lower().strip().rstrip("?")
    return qa.get(key, "Here's a healthy eating tip: Focus on whole foods and stay hydrated.")

def handle_fitness_tips(question: str) -> str:
    qa = {
        "what are the best exercises for beginners": "- Start with bodyweight exercises like squats, push-ups, and walking.",
        "how to build muscle": "- Combine strength training with proper nutrition, focusing on protein intake.",
        "best cardio for burning fat": "- Running, cycling, swimming, or HIIT workouts are effective.",
        "whats a good workout routine for a week": "- Alternate strength training and cardio; rest 1–2 days a week.",
        "how to stretch before a workout": "- Do dynamic stretches like leg swings, arm circles, and lunges.",
        "advice for staying motivated to exercise": "- Set goals, track progress, vary workouts, and find a workout buddy."
    }
    key = question.lower().strip().rstrip("?")
    return qa.get(key, "Here's a general tip: Stay consistent and listen to your body.")

def handle_recipes(question: str) -> str:
    qa = {
        "can you give me a healthy chicken recipe": "- Try grilled chicken with veggies and quinoa.",
        "whats a good salad recipe for lunch": "- Mix greens, cherry tomatoes, cucumber, grilled chicken, and olive oil.",
        "easy and healthy dinner ideas": "- Stir-fried veggies with tofu or salmon with steamed veggies.",
        "how to make a high-protein smoothie": "- Blend Greek yogurt, protein powder, banana, and berries."
    }
    key = question.lower().strip().rstrip("?")
    return qa.get(key, "Here's a healthy recipe tip: Include lean proteins and fresh vegetables.")

def handle_general_health(question: str) -> str:
    qa = {
        "how many hours of sleep do i need": "- Most adults need 7–9 hours of sleep per night.",
        "tips for managing stress": "- Practice mindfulness, exercise regularly, and maintain a balanced lifestyle.",
        "why is hydration important": "- It helps energy, brain function, and keeps your skin healthy."
    }
    key = question.lower().strip().rstrip("?")
    return qa.get(key, "Remember to stay active, hydrated, and sleep well.")

def handle_water_benefits(question: str) -> str:
    qa = {
        "what are the benefits of drinking water": "- Keeps you hydrated, aids digestion, maintains skin health, regulates temperature, lubricates joints, and flushes toxins.",
        "why should i drink water": "- Water is essential for energy, brain function, and overall health.",
        "is drinking water good for health": "- Absolutely, water supports all vital functions in your body."
    }
    key = question.lower().strip().rstrip("?")
    return qa.get(key, "Drink water regularly to stay healthy and hydrated.")

# ------------------- Gemini API Call -------------------
def get_api_response(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "API key is not configured. Please set GOOGLE_API_KEY in your .env file."
    try:
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

    print(f"Prompt: {prompt}")
    print(f"Predicted category: {category}")
    print(f"Confidence score: {max_prob}")

    CONFIDENCE_THRESHOLD = 0.3

    if max_prob > CONFIDENCE_THRESHOLD:
        if category == "eating_tips":
            response_text = handle_eating_tips(prompt)
        elif category == "fitness_tips":
            response_text = handle_fitness_tips(prompt)
        elif category == "recipes":
            response_text = handle_recipes(prompt)
        elif category == "general_health":
            response_text = handle_general_health(prompt)
        elif category == "water_benefits":
            response_text = handle_water_benefits(prompt)
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
