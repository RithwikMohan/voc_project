# ------------------- app.py -------------------

import os
import re
import math
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import nltk
from dotenv import load_dotenv
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception as e:
    genai = None
    GENAI_AVAILABLE = False
    print("⚠️ Warning: google.generativeai import failed:", e)

print("🔥 RUNNING app.py")

# -------- Load .env and configure Gemini --------
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if GENAI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print("⚠️ Warning: failed to configure generative client:", e)
        GENAI_AVAILABLE = False

# ------------------- NLTK Setup -------------------
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = stopwords.words("english")

# ------------------- Training Data (ML Classifier) -------------------
data = [
    ("What are some good high-protein foods?", "eating_tips"),
    ("Healthy breakfast ideas for weight loss", "eating_tips"),
    ("How to reduce sugar intake?", "eating_tips"),

    ("What are the best exercises for beginners?", "fitness_tips"),
    ("How to build muscle?", "fitness_tips"),
    ("How to stretch before a workout?", "fitness_tips"),

    ("Can you give me a healthy chicken recipe?", "recipes"),
    ("What's a good salad recipe?", "recipes"),

    ("How many hours of sleep do I need?", "general_health"),
    ("Tips for managing stress", "general_health"),

    ("What are the benefits of drinking water?", "water_benefits"),

    ("Create a health plan for me", "health_plan"),
    ("Calculate my calories", "health_plan"),
    ("Personalized fitness plan", "health_plan"),
    ("Make a diet plan", "health_plan"),
    ("Health assessment", "health_plan")
]

train_texts, train_labels = zip(*data)
vectorizer = TfidfVectorizer(stop_words=stop_words)
X_train = vectorizer.fit_transform(train_texts)
clf = LogisticRegression()
clf.fit(X_train, train_labels)

# ------------------- Conversation Memory -------------------
conversation_memory = []
MAX_MEMORY = 6

# ------------------- Step 2: Advanced Responses -------------------
ADVANCED_RESPONSES = {

    "diet_plan": """
**Daily Healthy Diet Plan**
• Breakfast: Oats + fruits + nuts  
• Mid-morning: 1 fruit  
• Lunch: 2 chapati + dal + salad  
• Snack: Dry fruits or green tea  
• Dinner: Soup + veggies + paneer/chicken  
• Water: 2–3 liters daily  
""",

    "workout_plan": """
**Beginner 1-Week Workout Plan**
• Mon: Full body workout  
• Tue: 20 min brisk walk  
• Wed: Core workout + stretching  
• Thu: Light cardio  
• Fri: Strength training  
• Sat: Yoga / mobility  
• Sun: Rest & hydration  
""",

    "high_protein_day": """
**High-Protein Day Plan**
• Breakfast: Eggs + oats + milk  
• Lunch: Chicken/paneer + veggies  
• Snack: Peanut butter sandwich  
• Dinner: Dal + sprouts + salad  
• Smoothie: Yogurt + banana + whey  
""",

    "weight_loss_plan": """
**Weight Loss Plan**
• 500 calorie deficit  
• 10,000 steps daily  
• Avoid sugar & junk  
• Add protein every meal  
• Sleep 7–9 hours  
• 20 min daily workout  
""",

    "weight_gain_plan": """
**Healthy Weight Gain Plan**
• Add 300–400 extra calories  
• Banana + peanut butter  
• Milk, curd, paneer, chicken  
• 4 days strength training  
• Protein: 1.6g/kg body weight  
"""
}

# ------------------- Health Recommendation Helpers -------------------
def is_health_profile(text: str) -> bool:
    """Rudimentary check whether the user supplied a health profile (age/weight/height).
    We look for keywords and numeric patterns (years, kg/lb, cm/m).
    """
    if not text or len(text) < 10:
        return False
    t = text.lower()
    # must contain at least one of these keywords or numeric markers
    if any(k in t for k in ("age", "weight", "height", "kg", "cm", "lbs", "lb")):
        return True
    # also accept patterns like '20, 70kg, 175cm' or 'i am 20, 70, 175'
    nums = re.findall(r"\b\d{2,3}(?:\.\d+)?\b", t)
    return len(nums) >= 2


def parse_health_details(text: str) -> dict:
    """Extract age, weight (kg), height (cm) and goal from free text.
    Returns a dict with keys: age, weight_kg, height_cm, goal.
    Units supported: kg, lbs, cm, m. If unit missing, heuristics apply.
    """
    t = text.lower()
    age = None
    weight_kg = None
    height_cm = None
    goal = None

    # extract age
    m = re.search(r"age\s*(?:is|:)?\s*(\d{1,3})", t)
    if m:
        age = int(m.group(1))
    else:
        m = re.search(r"(\d{1,3})\s*(?:years|yrs|y|yo)\b", t)
        if m:
            age = int(m.group(1))

    # extract weight
    m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(kg|kgs|kilograms)\b", t)
    if m:
        weight_kg = float(m.group(1))
    else:
        m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(lb|lbs)\b", t)
        if m:
            weight_kg = float(m.group(1)) * 0.45359237

    # extract height
    m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(cm|centimeters)\b", t)
    if m:
        height_cm = float(m.group(1))
    else:
        # meters (e.g., 1.75 m)
        m = re.search(r"(\d(?:\.\d+)?)\s*(m|meters)\b", t)
        if m:
            height_cm = float(m.group(1)) * 100.0
        else:
            # sometimes users write plain number for height (e.g., 175)
            m = re.findall(r"\b(1\d{2}|\d{2})\b", t)
            if m:
                # choose the largest plausible number for height
                cand = sorted(set([int(x) for x in m]), reverse=True)
                for c in cand:
                    if 100 <= c <= 230:
                        height_cm = float(c)
                        break

    # fallback: if user typed 3 numbers in sequence like '20, 70, 175'
    if (age is None or weight_kg is None or height_cm is None):
        nums = re.findall(r"\b\d{1,3}(?:\.\d+)?\b", t)
        if len(nums) >= 2:
            # heuristics: find plausible age (<= 120), weight (30-300), height (100-230)
            candidates = [float(n) for n in nums]
            possible_age = next((int(n) for n in candidates if 10 <= n <= 120), None)
            possible_height = next((n for n in reversed(candidates) if 100 <= n <= 230), None)
            possible_weight = next((n for n in candidates if 30 <= n <= 300 and n != possible_height), None)
            if age is None and possible_age is not None:
                age = int(possible_age)
            if height_cm is None and possible_height is not None:
                height_cm = float(possible_height)
            if weight_kg is None and possible_weight is not None:
                weight_kg = float(possible_weight)

    # detect goal
    if "lose" in t or "lose weight" in t or "weight loss" in t or "fat" in t:
        goal = "lose"
    elif "gain" in t or "gain weight" in t or "bulk" in t:
        goal = "gain"
    elif "maintain" in t or "maintain weight" in t or "maintaining" in t:
        goal = "maintain"

    return {
        "age": age,
        "weight_kg": round(weight_kg, 1) if weight_kg is not None else None,
        "height_cm": round(height_cm, 1) if height_cm is not None else None,
        "goal": goal or "maintain",
    }


def calculate_health_plan(details: dict) -> str:
    """Given parsed details, calculate BMI, calorie target, protein, water, and produce simple plans.
    Returns a formatted text response suitable for the chat UI.
    """
    age = details.get("age")
    w = details.get("weight_kg")
    h = details.get("height_cm")
    goal = details.get("goal", "maintain")

    if not (w and h and age):
        return "I couldn't extract all details. Please include your age, weight (kg or lbs) and height (cm or m). Example: 'My age is 20, weight 70kg, height 175cm. I want to lose weight.'"

    height_m = h / 100.0
    bmi = w / (height_m * height_m)

    # BMI category
    if bmi < 18.5:
        bmi_cat = "Underweight"
    elif bmi < 25:
        bmi_cat = "Normal"
    elif bmi < 30:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obese"

    # BMR estimate: average of male and female Mifflin-St Jeor (gender not provided)
    bmr_male = 10 * w + 6.25 * h - 5 * age + 5
    bmr_female = 10 * w + 6.25 * h - 5 * age - 161
    bmr = (bmr_male + bmr_female) / 2.0

    # activity factor default (moderate)
    activity_factor = 1.45
    maintenance_cal = bmr * activity_factor

    # calorie target based on goal
    if goal == "lose":
        calorie_target = max(1200, maintenance_cal - 500)
    elif goal == "gain":
        calorie_target = maintenance_cal + 300
    else:
        calorie_target = maintenance_cal

    # protein target (range) in grams per kg
    if goal == "lose":
        prot_min = 1.6
        prot_max = 2.2
    elif goal == "gain":
        prot_min = 1.6
        prot_max = 2.0
    else:
        prot_min = 1.2
        prot_max = 1.6

    protein_min_g = math.ceil(prot_min * w)
    protein_max_g = math.ceil(prot_max * w)

    # water intake (approx 35 ml per kg)
    water_l = round(w * 0.035, 2)

    # Simple beginner workout
    workout = (
        "Beginner Workout Plan:\n"
        "• 10 min brisk walk (warm-up)\n"
        "• Bodyweight squats: 3 sets × 12 reps\n"
        "• Push-ups (knees if needed): 3 sets × 8–12 reps\n"
        "• Plank: 3 × 20–40 sec\n"
        "• Stretching/cool-down: 5–10 min\n"
    )

    # Meal plan (very rough distribution)
    cal = int(round(calorie_target))
    breakfast = int(round(cal * 0.25))
    lunch = int(round(cal * 0.35))
    snack = int(round(cal * 0.10))
    dinner = cal - (breakfast + lunch + snack)

    meal_plan = (
        "Suggested Daily Meal Plan:\n"
        f"• Breakfast ({breakfast} kcal): Oats with milk, 1 banana, handful of nuts (protein ~15–20g)\n"
        f"• Lunch ({lunch} kcal): 2 chapatis or rice, dal or chicken, mixed veggies (protein ~25–35g)\n"
        f"• Snack ({snack} kcal): Greek yogurt or fruit + handful of almonds (protein ~8–12g)\n"
        f"• Dinner ({dinner} kcal): Roti + vegetable curry + salad (protein ~20–30g)\n"
    )

    resp = (
        f"Your BMI: {bmi:.1f} ({bmi_cat})\n"
        f"Estimated maintenance calories: ~{int(round(maintenance_cal))} kcal/day\n"
        f"Recommended daily calories ({goal}): ~{cal} kcal/day\n"
        f"Protein target: {protein_min_g}–{protein_max_g} g/day\n"
        f"Water: ~{water_l} L/day\n\n"
        f"{workout}\n"
        f"{meal_plan}\n"
        "Notes: These are general guidelines (not medical advice). If you can, provide sex and activity level for a tighter estimate."
    )

    return resp


# ------------------- Prompt Engineering -------------------
SYSTEM_PROMPT = """
You are a helpful Health, Fitness & Nutrition AI Assistant.
Rules:
1. No medical diagnosis.
2. Keep responses short, clear, and safe.
3. Use bullet points when possible.
4. Be friendly and supportive.
5. If question is not related to health/fitness/nutrition/recipes, politely refuse.
"""

FEW_SHOTS = """
User: Suggest a healthy breakfast.
AI: • Oats with fruits  
    • Nuts  
    • A boiled egg  

User: Give me a beginner workout.
AI: • 10 min warm-up  
    • 10 squats  
    • 10 pushups  
"""

def get_api_response(prompt: str) -> str:
    if not GEMINI_API_KEY or not GENAI_AVAILABLE:
        return "LLM unavailable (API key or client missing). I can still help with basic tips or use the Health Plan button for structured recommendations."

    # include conversation memory
    memory_text = "\n".join(conversation_memory[-MAX_MEMORY:])

    final_prompt = f"""
{SYSTEM_PROMPT}

{FEW_SHOTS}

Conversation:
{memory_text}

User: {prompt}
AI:
"""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(final_prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini Error:", e)
        return "Sorry, I couldn't fetch the answer from the LLM. Try again or use the Health Plan form for structured help."

# ------------------- Flask App -------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/test")
def test():
    return "<h1>Test Route Works!</h1><p>Flask is working correctly.</p>"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    raw_prompt = (data.get("prompt") or "").strip()
    prompt = raw_prompt.lower()

    print(f"🔥 Received prompt: {raw_prompt}")  # Debug log

    if not raw_prompt:
        return jsonify({"response": "Please type something."})

    # add user message to memory (store the raw user text for clarity)
    conversation_memory.append(f"User: {raw_prompt}")

    # Enhanced specific responses first (before ML classifier)
    # Weight gain advice
    if "gain weight" in prompt or "weight gain" in prompt:
        response = """**Healthy Weight Gain Plan**
• Add 300–400 extra calories daily
• Eat protein-rich foods: eggs, chicken, paneer, fish
• Include healthy fats: nuts, avocado, olive oil
• Strength training 3-4 times per week
• Eat frequent small meals (5-6 per day)
• Drink milk and protein smoothies
• Get 7-9 hours quality sleep
• Stay hydrated with water"""
        conversation_memory.append(f"AI: {response}")
        return jsonify({"response": response})
    
    # Chicken recipe
    elif "chicken" in prompt and "recipe" in prompt:
        response = """**Healthy Chicken Recipe**
• Chicken breast: 200g (boneless)
• Marinate: yogurt, ginger-garlic paste, spices (1 hour)
• Cooking: Grill or bake at 180°C for 20-25 minutes
• Seasoning: Salt, pepper, oregano, thyme
• Serve with: Brown rice and steamed vegetables
• Garnish: Fresh herbs and lemon juice
• Cooking time: 30 minutes total"""
        conversation_memory.append(f"AI: {response}")
        return jsonify({"response": response})

    # If this looks like a health-profile (age/weight/height) -> compute plan
    if is_health_profile(raw_prompt):
        details = parse_health_details(raw_prompt)
        response = calculate_health_plan(details)
        conversation_memory.append(f"AI: {response}")
        return jsonify({"response": response})

    # ---------------- STEP 2: ADVANCED RESPONSE TRIGGERS ----------------
    if "diet" in prompt or "meal plan" in prompt:
        response = ADVANCED_RESPONSES["diet_plan"]
        conversation_memory.append(f"AI: {response}")
        return jsonify({"response": response})

    if "workout" in prompt or "routine" in prompt:
        response = ADVANCED_RESPONSES["workout_plan"]
        conversation_memory.append(f"AI: {response}")
        return jsonify({"response": response})

    if "protein" in prompt or "high protein" in prompt:
        response = ADVANCED_RESPONSES["high_protein_day"]
        conversation_memory.append(f"AI: {response}")
        return jsonify({"response": response})

    if "weight loss" in prompt or "lose weight" in prompt:
        response = ADVANCED_RESPONSES["weight_loss_plan"]
        conversation_memory.append(f"AI: {response}")
        return jsonify({"response": response})

    # ---------------- ML CLASSIFIER ----------------
    X_test = vectorizer.transform([prompt])
    probs = clf.predict_proba(X_test)[0]
    max_prob = float(max(probs))
    category = clf.predict(X_test)[0]

    CONFIDENCE_THRESHOLD = 0.32

    if max_prob > CONFIDENCE_THRESHOLD:
        if category == "eating_tips":
            response = "- Eat whole foods\n- Add protein\n- Reduce sugar\n- Stay hydrated"
        elif category == "fitness_tips":
            response = "- Walk daily\n- Do squats\n- Do pushups\n- Stretch before workouts"
        elif category == "recipes":
            response = "- Try quinoa + veggies\n- Fruit smoothie\n- Chicken/paneer bowl"
        elif category == "general_health":
            response = "- Sleep 7–9 hours\n- Drink water\n- Reduce screen time"
        elif category == "water_benefits":
            response = "- Hydration boosts energy, digestion, skin health"
        elif category == "health_plan":
            response = "To create a personalized health plan, please click the 'Health Plan' button above or provide your age, weight, height, and goal (e.g., 'age 25, weight 70kg, height 175cm, lose weight')."
        else:
            response = get_api_response(prompt)
    else:
        response = get_api_response(prompt)

    # add AI answer to memory
    conversation_memory.append(f"AI: {response}")

    return jsonify({"response": response})


@app.route("/health_plan", methods=["POST"])
def health_plan():
    data = request.get_json(silent=True) or {}
    # Expected fields: age, sex (opt), weight, weight_unit, height, height_unit, activity, goal
    try:
        age = int(data.get('age')) if data.get('age') is not None else None
    except Exception:
        age = None

    sex = data.get('sex') or None

    try:
        weight = float(data.get('weight')) if data.get('weight') is not None else None
    except Exception:
        weight = None
    weight_unit = (data.get('weight_unit') or 'kg').lower()

    try:
        height = float(data.get('height')) if data.get('height') is not None else None
    except Exception:
        height = None
    height_unit = (data.get('height_unit') or 'cm').lower()

    goal = (data.get('goal') or '').lower() or 'maintain'

    # Normalize units
    if weight is not None and weight_unit in ('lb', 'lbs'):
        weight_kg = round(weight * 0.45359237, 1)
    else:
        weight_kg = round(weight, 1) if weight is not None else None

    if height is not None and height_unit in ('m', 'meter', 'meters'):
        height_cm = round(height * 100.0, 1)
    else:
        height_cm = round(height, 1) if height is not None else None

    details = {
        'age': age,
        'weight_kg': weight_kg,
        'height_cm': height_cm,
        'goal': goal,
        'sex': sex
    }

    response = calculate_health_plan(details)
    conversation_memory.append(f"AI: {response}")
    return jsonify({'response': response})

@app.route("/health")
def health():
    return jsonify({"ok": True, "gemini_key_loaded": bool(GEMINI_API_KEY)})

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(use_reloader=False)
