import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ------------------------------------------------
# 1️⃣ Flask Setup
# ------------------------------------------------
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "skincare_model.pth"

# ------------------------------------------------
# 2️⃣ Skin Disease Classes (31)
# ------------------------------------------------
classes = [
    "Acne", "Actinic Keratosis", "Alopecia Areata", "Atopic Dermatitis",
    "Basal Cell Carcinoma", "Bullous Disease", "Cellulitis", "Eczema",
    "Folliculitis", "Herpes Simplex", "Impetigo", "Lichen Planus",
    "Lupus Erythematosus", "Melanoma", "Molluscum Contagiosum", "Nevus",
    "Onychomycosis", "Pityriasis Rosea", "Psoriasis", "Rosacea",
    "Scabies", "Seborrheic Keratosis", "Shingles (Herpes Zoster)",
    "Squamous Cell Carcinoma", "Tinea Corporis", "Tinea Versicolor",
    "Urticaria (Hives)", "Varicella (Chickenpox)", "Vasculitis",
    "Vitiligo", "Warts"
]

# ------------------------------------------------
# 3️⃣ Recommendations for Each Class
# ------------------------------------------------
recommendations = {
    "Acne": ["Salicylic Acid Cleanser", "Niacinamide Serum", "Oil-Free Moisturizer"],
    "Actinic Keratosis": ["Use SPF 50+ sunscreen", "Consult a dermatologist", "Topical 5-fluorouracil cream"],
    "Alopecia Areata": ["Minoxidil (Rogaine)", "Corticosteroid cream", "Biotin supplements"],
    "Atopic Dermatitis": ["CeraVe Healing Ointment", "Eucerin Eczema Relief", "Avoid fragrance products"],
    "Basal Cell Carcinoma": ["Seek medical treatment", "Photodynamic therapy", "Use sunscreen daily"],
    "Bullous Disease": ["Topical steroids", "Antibiotic cream (if infected)", "Consult dermatologist"],
    "Cellulitis": ["Antibiotic medication", "Keep skin clean", "Seek medical advice immediately"],
    "Eczema": ["Colloidal oatmeal lotion", "Hydrocortisone cream", "Moisturize frequently"],
    "Folliculitis": ["Benzoyl peroxide wash", "Avoid tight clothing", "Warm compress for relief"],
    "Herpes Simplex": ["Acyclovir cream", "Keep affected area clean", "Avoid touching blisters"],
    "Impetigo": ["Antibacterial cream", "Wash skin gently", "Consult doctor for antibiotics"],
    "Lichen Planus": ["Topical corticosteroids", "Aloe vera gel", "Avoid scratching"],
    "Lupus Erythematosus": ["Sunscreen SPF 50+", "Anti-inflammatory medications", "Consult rheumatologist"],
    "Melanoma": ["Seek immediate oncologist consultation", "Avoid sun exposure", "Regular skin checks"],
    "Molluscum Contagiosum": ["Cryotherapy", "Topical retinoids", "Keep area dry and clean"],
    "Nevus": ["Monitor mole for changes", "Use sunscreen", "Consult dermatologist for removal if needed"],
    "Onychomycosis": ["Antifungal nail cream", "Keep nails short and dry", "Avoid nail polish"],
    "Pityriasis Rosea": ["Calamine lotion", "Hydrocortisone cream", "Gentle soap and moisturizer"],
    "Psoriasis": ["Coal tar shampoo", "Vitamin D ointment", "Fragrance-free moisturizers"],
    "Rosacea": ["Gentle cleanser", "Azelaic acid gel", "Avoid spicy foods and alcohol"],
    "Scabies": ["Permethrin cream", "Wash bedding in hot water", "Consult a doctor"],
    "Seborrheic Keratosis": ["Cryotherapy or laser treatment", "Moisturize skin", "Consult dermatologist if inflamed"],
    "Shingles (Herpes Zoster)": ["Antiviral medication (acyclovir)", "Pain relief cream", "Cool compress"],
    "Squamous Cell Carcinoma": ["Medical surgery required", "Avoid sunlight", "Follow up with dermatologist"],
    "Tinea Corporis": ["Antifungal cream (clotrimazole)", "Keep area dry", "Avoid sharing towels"],
    "Tinea Versicolor": ["Selenium sulfide shampoo", "Antifungal soap", "Moisturize gently"],
    "Urticaria (Hives)": ["Antihistamine tablet", "Calamine lotion", "Avoid allergens"],
    "Varicella (Chickenpox)": ["Calamine lotion", "Oatmeal bath", "Hydration and rest"],
    "Vasculitis": ["Medical evaluation required", "Anti-inflammatory drugs", "Avoid infections"],
    "Vitiligo": ["Topical corticosteroids", "Vitamin B12 and folic acid", "Sunscreen SPF 50+"],
    "Warts": ["Salicylic acid patches", "Cryotherapy", "Avoid touching warts"]
}

# ------------------------------------------------
# 4️⃣ Load Model
# ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

try:
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ------------------------------------------------
# 5️⃣ Image Transform
# ------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------------------------
# 6️⃣ Prediction Function
# ------------------------------------------------
def predict_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
            _, predicted = outputs.max(1)
            return int(predicted.item())
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return None

# ------------------------------------------------
# 7️⃣ Flask Endpoint
# ------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        pred_idx = predict_image(file_path)
        if pred_idx is None or pred_idx >= len(classes):
            return jsonify({"error": "Prediction failed"}), 500

        predicted_label = classes[pred_idx]
        rec = recommendations.get(predicted_label, ["Consult a dermatologist for treatment."])

        return jsonify({
            "result": predicted_label,
            "recommendations": rec
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------
# 8️⃣ Run App
# ------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
