import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import time

# ─── Configuration de la page ───────────────────────────────────────────────
st.set_page_config(
    page_title="African Food Classifier",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── CSS personnalisé ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Lato:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Lato', sans-serif;
    }

    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    .result-card {
        background: linear-gradient(135deg, #f8f4e3 0%, #fef9ec 100%);
        border-left: 5px solid #e8a020;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(232, 160, 32, 0.15);
    }

    .result-food {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: #c17f10;
        margin: 0;
    }

    .result-confidence {
        font-size: 1.1rem;
        color: #555;
        margin-top: 0.3rem;
    }

    .food-info {
        background: #fff;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
        border: 1px solid #eee;
        font-size: 0.95rem;
        color: #444;
    }

    .top3-item {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #f0f0f0;
        font-size: 0.9rem;
    }

    .badge {
        background: #e8a020;
        color: white;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .stProgress > div > div > div > div {
        background-color: #e8a020;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Informations sur les plats ──────────────────────────────────────────────
FOOD_INFO = {
    "Ekwang": {
        "pays": "🇨🇲 Cameroun",
        "description": "Plat de feuilles de macabo farcies de tubercules râpés et de viande, mijotées à l'huile de palme.",
        "emoji": "🥬"
    },
    "Eru": {
        "pays": "🇨🇲 Cameroun",
        "description": "Légumes de forêt (Gnetum africanum) cuits avec de l'huile de palme et accompagnés de water-fufu.",
        "emoji": "🌿"
    },
    "Ndole": {
        "pays": "🇨🇲 Cameroun",
        "description": "Plat national camerounais à base de feuilles amères, de crevettes et d'arachides. Incontournable !",
        "emoji": "🍲"
    },
    "Jollof Rice": {
        "pays": "🇬🇭 Ghana / Afrique de l'Ouest",
        "description": "Riz cuit dans une sauce tomate épicée, emblème de la gastronomie ouest-africaine.",
        "emoji": "🍚"
    },
    "Palm-nut Soup": {
        "pays": "🇬🇭 Ghana",
        "description": "Soupe riche à base de pulpe de noix de palme, servie avec du fufu ou du riz.",
        "emoji": "🥘"
    },
    "Waakye": {
        "pays": "🇬🇭 Ghana",
        "description": "Mélange de riz et haricots cuits ensemble, plat populaire de rue servi avec divers accompagnements.",
        "emoji": "🫘"
    }
}

# ─── Chargement du modèle ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('african_food_model.h5')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

# ─── Fonction de prédiction ──────────────────────────────────────────────────
def predict(image_pil, model, class_names):
    img = image_pil.resize((224, 224))
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:  # RGBA → RGB
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)[0]
    top3_idx = np.argsort(predictions)[::-1][:3]
    return [
        {"class": class_names[i], "confidence": float(predictions[i])}
        for i in top3_idx
    ]

# ─── Interface principale ────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🍽️ African Food Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Reconnaissance de plats africains par intelligence artificielle — M2 IABD</p>', unsafe_allow_html=True)

st.markdown("---")

# Chargement du modèle
with st.spinner("Chargement du modèle..."):
    try:
        model, class_names = load_model()
        st.success(f" Modèle chargé — {len(class_names)} classes détectées")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.info("Assure-toi que `african_food_model.h5` et `class_names.json` sont dans le même dossier que `app.py`.")
        st.stop()

# Upload de l'image
st.markdown("### 📸 Uploader une image")
uploaded_file = st.file_uploader(
    "Glisse ou clique pour uploader une photo de plat",
    type=["jpg", "jpeg", "png", "webp"],
    help="Formats acceptés : JPG, PNG, WEBP"
)

if uploaded_file:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="Image uploadée", width="stretch")

    with col2:
        with st.spinner("Analyse en cours..."):
            time.sleep(0.5)
            results = predict(image_pil, model, class_names)

        top = results[0]
        food_name = top["class"]
        confidence = top["confidence"]

        # Résultat principal
        st.markdown(f"""
        <div class="result-card">
            <p class="result-food">{FOOD_INFO.get(food_name, {}).get('emoji', '🍽️')} {food_name}</p>
            <p class="result-confidence">Confiance : <strong>{confidence*100:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Barre de confiance
        st.progress(confidence)

        # Infos sur le plat
        if food_name in FOOD_INFO:
            info = FOOD_INFO[food_name]
            st.markdown(f"""
            <div class="food-info">
                <strong>{info['pays']}</strong><br>
                {info['description']}
            </div>
            """, unsafe_allow_html=True)

        # Top 3
        st.markdown("#### 📊 Top 3 des prédictions")
        for i, res in enumerate(results):
            medal = ["🥇", "🥈", "🥉"][i]
            st.markdown(f"""
            <div class="top3-item">
                <span>{medal} {res['class']}</span>
                <span class="badge">{res['confidence']*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

else:
    # Placeholder
    st.info("👆 Uploade une image pour commencer l'analyse")

    st.markdown("### 🌍 Plats reconnus par le modèle")
    cols = st.columns(3)
    foods = list(FOOD_INFO.items())
    for i, (name, info) in enumerate(foods):
        with cols[i % 3]:
            st.markdown(f"""
            **{info['emoji']} {name}**  
            {info['pays']}  
            <small style='color:#888'>{info['description'][:60]}...</small>
            """, unsafe_allow_html=True)
            st.markdown("")

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "African Food Classifier • M2 IABD • Transfer Learning MobileNetV2 • TensorFlow"
    "</div>",
    unsafe_allow_html=True
)
