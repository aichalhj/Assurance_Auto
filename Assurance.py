#%%
# pages/1_Modélisation_tarification.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.datasets import fetch_openml

st.set_page_config(page_title="Modélisation & Simulation", layout="wide")
st.title("🧮 Modélisation et estimation de la Prime Pure")

# --- Chargement des données ---
@st.cache_data
def load_data():
    freq = fetch_openml("freMTPL2freq", version=1, as_frame=True).frame
    sev = fetch_openml("freMTPL2sev", version=1, as_frame=True).frame
    data = freq.merge(
        sev.groupby("IDpol")["ClaimAmount"].sum().reset_index(),
        on="IDpol", how="left"
    )
    data["ClaimAmount"] = data["ClaimAmount"].fillna(0)
    data["Exposure"] = data["Exposure"].replace(0, 1e-6)  # éviter les logs 0
    return data

data = load_data()

# --- Préparation des classes ---
# Tranches d'âge du conducteur
bins_age = [17, 20, 30, 40, 50, 60, 70, 80, 120]
labels_age = ['18-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+']
data['DrivAge_class'] = pd.cut(data['DrivAge'], bins=bins_age, labels=labels_age, right=True)

# Tranches CV fiscales (14 classes)
bins_power = [0, 4, 6, 8, 10, 12, 15, 18, 21, 25, 30, 40, 60, 80, 100]
labels_power = [
    '1-4 CV', '5-6 CV', '7-8 CV', '9-10 CV', '11-12 CV', 
    '13-15 CV', '16-18 CV', '19-21 CV', '22-25 CV', 
    '26-30 CV', '31-40 CV', '41-60 CV', '61-80 CV', '81+ CV'
]
data['VehPower_class'] = pd.cut(data['VehPower'], bins=bins_power, labels=labels_power, right=True)

# --- Modèle fréquence (Poisson) ---
formula_freq = """
ClaimNb ~ VehPower_class + VehAge + DrivAge_class
+ BonusMalus + Area + VehBrand + VehGas + Density + Region
"""
glm_freq = smf.glm(
    formula=formula_freq,
    data=data,
    family=sm.families.Poisson(),
    offset=np.log(data["Exposure"])
).fit()

# --- Modèle sévérité (Gamma) ---
data_sev = data[data["ClaimAmount"] > 0].copy()
data_sev["Severity"] = data_sev["ClaimAmount"] / data_sev["ClaimNb"]

formula_sev = """
Severity ~ VehPower_class + VehAge + DrivAge_class
+ BonusMalus + Area + VehBrand + VehGas + Density + Region
"""
glm_sev = smf.glm(
    formula=formula_sev,
    data=data_sev,
    family=sm.families.Gamma(sm.families.links.log())
).fit()

st.success("Modèles entraînés avec succès (fréquence + sévérité)")

# --- Interface utilisateur pour simuler un contrat ---
st.subheader("🧍‍♂️ Saisissez les caractéristiques du contrat")

col1, col2, col3 = st.columns(3)

with col1:
    veh_age = st.slider(
    "Âge du véhicule (années)",
    int(data["VehAge"].min()),
    int(data["VehAge"].max()),
    int(data["VehAge"].median())
)

    veh_power_class = st.selectbox(
        "Tranche de puissance du véhicule",
        labels_power
    )
    veh_brand = st.selectbox("Marque du véhicule", sorted(data["VehBrand"].unique()))

with col2:
    driv_age = st.slider("Âge du conducteur", 18, 90, 40)
    driv_age_class = pd.cut([driv_age], bins=bins_age, labels=labels_age)[0]
    bonus_malus = st.slider("Bonus-Malus", int(data["BonusMalus"].min()), int(data["BonusMalus"].max()), 100)
    veh_gas = st.selectbox("Type de carburant", sorted(data["VehGas"].unique()))

with col3:
    area = st.selectbox("Zone", sorted(data["Area"].unique()))
    region = st.selectbox("Région", sorted(data["Region"].unique()))
    density = st.number_input("Densité (habitants/km²)", min_value=0, max_value=50000, value=3000)
    exposure = st.number_input("Durée d’exposition (années)", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

# --- Créer un DataFrame pour la prédiction ---
contract = pd.DataFrame({
    "VehPower_class": [veh_power_class],
    "VehAge": [veh_age],
    "DrivAge_class": [driv_age_class],
    "BonusMalus": [bonus_malus],
    "Area": [area],
    "VehBrand": [veh_brand],
    "VehGas": [veh_gas],
    "Density": [density],
    "Region": [region],
    "Exposure": [exposure]
})

# --- Prédiction fréquence + sévérité ---
try:
    contract["ExpectedClaims"] = glm_freq.predict(contract)
    contract["ExpectedSeverity"] = glm_sev.predict(contract)
    contract["PurePremium"] = contract["ExpectedClaims"] * contract["ExpectedSeverity"]

    st.markdown("---")
    st.subheader("📊 Résultats du calcul de prime pure")
    st.write(f"""
    - **Fréquence attendue (sinistres/an)** : {contract['ExpectedClaims'].iloc[0]:.4f}  
    - **Sévérité attendue (€ / sinistre)** : {contract['ExpectedSeverity'].iloc[0]:,.0f} €  
    - **Prime pure estimée (€ / an)** : **{contract['PurePremium'].iloc[0]:,.0f} €**
    """)

    # Progress bar proportionnelle à 2000 € (à ajuster si nécessaire)
    st.progress(min(contract["PurePremium"].iloc[0] / 2000, 1.0))
except Exception as e:
    st.error(f"Erreur lors de la prédiction : {e}")

#%%
data.describe()
# %%
