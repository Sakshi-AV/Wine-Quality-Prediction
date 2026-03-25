import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Wine Quality AI", layout="wide")

# ===============================
# 🌌 CUSTOM AURORA GLASS UI
# ===============================
st.markdown("""
<style>

/* Animated Aurora Background */
body {
    background: linear-gradient(-45deg, #2b0f3a, #41295a, #1f4037, #3a1c71);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass Card */
.glass {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(18px);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.2);
    margin-bottom: 20px;
}

/* Title */
.title {
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:white;
}

/* Button Styling */
.stButton>button {
    background: linear-gradient(45deg, #ff0080, #7928ca);
    color: white;
    border-radius: 25px;
    padding: 10px 25px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #ff0080;
}

/* Make sliders prettier */
.css-1cpxqw2, .css-1d391kg {
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATA & MODEL
# ===============================
model = joblib.load("model.pkl")
df = pd.read_csv("winequality.csv.xls").dropna()
df["type"] = df["type"].map({"red":0,"white":1})

# ===============================
# HEADER
# ===============================
st.markdown("<div class='title'>🍷 AI Wine Quality Prediction</div>", unsafe_allow_html=True)
st.write("")

tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Visualizations"])

# ==========================================
# 🔮 PREDICTION TAB
# ==========================================
with tab1:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.sidebar.header("🍷 Wine Parameters")

    wine_type = st.sidebar.selectbox("🍇 Wine Type", ["red","white"])
    wine_type_num = 0 if wine_type=="red" else 1

    features = {
        "type": wine_type_num,
        "fixed acidity": st.sidebar.slider("🧪 Fixed Acidity",4.0,16.0,8.0),
        "volatile acidity": st.sidebar.slider("💨 Volatile Acidity",0.1,1.5,0.5),
        "citric acid": st.sidebar.slider("🍋 Citric Acid",0.0,1.0,0.3),
        "residual sugar": st.sidebar.slider("🍬 Residual Sugar",0.5,20.0,2.0),
        "chlorides": st.sidebar.slider("🧂 Chlorides",0.01,0.2,0.05),
        "free sulfur dioxide": st.sidebar.slider("🌫 Free Sulfur Dioxide",1,80,20),
        "total sulfur dioxide": st.sidebar.slider("☁ Total Sulfur Dioxide",6,300,100),
        "density": st.sidebar.slider("⚖ Density",0.990,1.005,0.996),
        "pH": st.sidebar.slider("🧬 pH Level",2.8,4.2,3.3),
        "sulphates": st.sidebar.slider("🧴 Sulphates",0.2,2.0,0.6),
        "alcohol": st.sidebar.slider("🍺 Alcohol %",8.0,15.0,10.5),
    }

    input_df = pd.DataFrame([features])

    if st.button("Predict Wine Quality 🍷"):

        pred = model.predict(input_df)[0]

        if pred=="Low":
            st.error("🍷 Low Quality Wine")
        elif pred=="Medium":
            st.warning("🍷 Medium Quality Wine")
        else:
            st.success("👑 Premium Quality Wine")

        st.write("### 📋 Input Summary")
        st.dataframe(input_df)

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 📊 VISUALIZATION TAB
# ==========================================
with tab2:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Small Quality Histogram
    st.subheader("Quality Distribution")
    fig1 = plt.figure(figsize=(4,3))
    df["quality"].hist(bins=10)
    plt.xlabel("Quality")
    plt.ylabel("Count")
    st.pyplot(fig1)

    # Small Heatmap
    st.subheader("Feature Correlation")
    fig2 = plt.figure(figsize=(6,4))
    sns.heatmap(df.corr(), cmap="coolwarm")
    st.pyplot(fig2)

    # Small Alcohol Plot
    st.subheader("Alcohol vs Quality")
    fig3 = plt.figure(figsize=(4,3))
    sns.boxplot(x="quality", y="alcohol", data=df)
    st.pyplot(fig3)

    # Feature Importance
    st.subheader("Model Feature Importance")
    try:
        importances = model.named_steps["rf"].feature_importances_
        cols = input_df.columns

        imp_df = pd.DataFrame({
            "Feature":cols,
            "Importance":importances
        }).sort_values(by="Importance",ascending=False)

        fig4 = plt.figure(figsize=(5,3))
        sns.barplot(x="Importance", y="Feature", data=imp_df)
        st.pyplot(fig4)

    except:
        st.info("Feature importance unavailable.")

    st.markdown("</div>", unsafe_allow_html=True)