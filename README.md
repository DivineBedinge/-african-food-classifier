# 🍽️ African Food Classifier — M2 IABD

Reconnaissance de plats africains par Transfer Learning (MobileNetV2 + TensorFlow).

## 📁 Structure du projet

```
african_food_classifier/
├── train_model.ipynb       ← Notebook Google Colab (entraînement)
├── app.py                  ← Application Streamlit
├── african_food_model.h5   ← Modèle entraîné (généré par le notebook)
├── class_names.json        ← Classes du modèle (généré par le notebook)
├── requirements.txt        ← Dépendances Python
├── Dockerfile              ← Pour déploiement Docker / Render
├── render_link.txt         ← Lien de l'app déployée sur Render
└── README.md
```

## 🌍 Dataset

- **Source :** [African Foods Datasets — Kaggle](https://www.kaggle.com/datasets/graccy/african-foods-datasets)
- **Classes :** Ekwang, Eru, Ndole (Cameroun) + Jollof Rice, Palm-nut Soup, Waakye (Ghana)
- **Taille :** ~1 754 images totales

## 🚀 Étapes

### 1. Entraînement (Google Colab)
1. Ouvre `train_model.ipynb` dans Google Colab
2. Active le GPU : `Exécution > Modifier le type d'exécution > GPU T4`
3. Exécute toutes les cellules
4. Télécharge `african_food_model.h5` et `class_names.json`
5. Place-les dans ce dossier

### 2. Test local
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3. Déploiement sur Render
1. Push le projet sur GitHub
2. Va sur [render.com](https://render.com) → New Web Service
3. Connecte ton repo GitHub
4. Configure :
   - **Build Command :** `pip install -r requirements.txt`
   - **Start Command :** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Copie le lien généré dans `render_link.txt`

## 🧠 Architecture du modèle

- **Base :** MobileNetV2 pré-entraîné sur ImageNet
- **Phase 1 :** Entraînement du classifier (base gelée) — 10 epochs
- **Phase 2 :** Fine-tuning des 30 dernières couches — 15 epochs
- **Accuracy cible :** > 85%

## 📦 Technologies

| Outil | Usage |
|---|---|
| TensorFlow / Keras | Modèle deep learning |
| MobileNetV2 | Transfer learning |
| Streamlit | Interface web |
| Render | Déploiement cloud |
| Docker | Conteneurisation |
| Google Colab | Entraînement GPU |
