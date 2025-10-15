# app.py
import re
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.datasets import fetch_openml

# -----------------------------
# CONFIGURATION PAGE
# -----------------------------
st.set_page_config(page_title="Tarification Automobile", layout="wide")

# -----------------------------
# CHARGEMENT DES DONNÉES
# -----------------------------
@st.cache_data
def load_data():
    freq = fetch_openml("freMTPL2freq", version=1, as_frame=True).frame
    sev = fetch_openml("freMTPL2sev", version=1, as_frame=True).frame
    data = freq.merge(
        sev.groupby("IDpol")["ClaimAmount"].sum().reset_index(),
        on="IDpol", how="left"
    )
    data["ClaimAmount"] = data["ClaimAmount"].fillna(0)
    data["ClaimFrequency"] = data["ClaimNb"] / data["Exposure"]
    data["Severity"] = np.where(data["ClaimNb"] > 0, data["ClaimAmount"] / data["ClaimNb"], 0)
    data["Exposure"] = data["Exposure"].replace(0, 1e-6)
    return data

data = load_data()

# -----------------------------
# Palette et style graphique
# -----------------------------
PALETTE = sns.color_palette("Set2")
COLOR_PRIMARY = PALETTE[1] if len(PALETTE) > 1 else PALETTE[0]
sns.set_style("whitegrid")
sns.set_palette(PALETTE)
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# -----------------------------
# MENU LATÉRAL
# -----------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Menu principal", "Analyse exploratoire", "Modélisation & Simulation"]
)

# =============================================================================
# PAGE 1 : MENU PRINCIPAL
# =============================================================================
if menu == "Menu principal":
    st.title("Tarification Automobile")
    st.markdown(
    """
    Pour cette étude, nous utilisons les jeux de données **freMTPL2freq** et **freMTPL2sev**, très utilisés en actuariat automobile.  
    Ils contiennent les caractéristiques des véhicules, la période d'exposition des contrats (par exemple, 100 jours correspondent à 100/365 ≈ 0,27 année), et les informations sur les conducteurs (âge, bonus-malus, etc.).  
    Ces données servent de base à toutes les analyses.

    L’application propose deux grandes parties :  

    - **Analyse exploratoire** : permet d’observer la fréquence et le montant des sinistres selon différents critères (région, véhicule, carburant, etc.) grâce à une interface interactive.  

    - **Modélisation & Simulation** : le modèle fonctionne en arrière-plan. L’utilisateur peut saisir les caractéristiques du conducteur, du véhicule, le bonus-malus, ainsi que la zone géographique et la densité. L’application fournit alors l’estimation de la fréquence des sinistres et du montant attendu pour ce contrat.

    Utilisez le menu à gauche pour naviguer entre les sections.
    """,
    unsafe_allow_html=True
)

# =============================================================================
# PAGE 2 : ANALYSE EXPLORATOIRE
# =============================================================================
elif menu == "Analyse exploratoire":
    st.title("Analyse exploratoire des sinistres")
    st.markdown("Explorez les données selon différents filtres.")

    st.sidebar.markdown("### Filtres")

    vehage_min, vehage_max = st.sidebar.slider(
        "Âge du véhicule", int(data["VehAge"].min()), int(data["VehAge"].max()),
        (int(data["VehAge"].min()), int(data["VehAge"].max()))
    )
    drivage_min, drivage_max = st.sidebar.slider(
        "Âge du conducteur", int(data["DrivAge"].min()), int(data["DrivAge"].max()),
        (int(data["DrivAge"].min()), int(data["DrivAge"].max()))
    )
    bonus_min, bonus_max = st.sidebar.slider(
        "Bonus-Malus", int(data["BonusMalus"].min()), int(data["BonusMalus"].max()),
        (int(data["BonusMalus"].min()), int(data["BonusMalus"].max()))
    )
    st.sidebar.markdown("---")

    def safe_key(column, val):
        return re.sub(r"\W+", "_", f"{column}_{val}")

    def n_columns_for(n_items):
        if n_items > 40:
            return 4
        elif n_items > 20:
            return 3
        elif n_items > 8:
            return 2
        return 1

    if "reset_filters" not in st.session_state:
        st.session_state["reset_filters"] = False

    if st.sidebar.button("Réinitialiser tous les filtres"):
        st.session_state["reset_filters"] = True

    def checkbox_grid_filter(column, label):
        values = sorted(data[column].astype(str).unique())
        ncols = n_columns_for(len(values))
        with st.sidebar.expander(label, expanded=False):
            for val in values:
                key = safe_key(column, val)
                if key not in st.session_state or st.session_state["reset_filters"]:
                    st.session_state[key] = True

            c1, c2 = st.columns(2)
            with c1:
                if st.button(f"Tout sélectionner", key=f"select_all_{column}"):
                    for val in values:
                        st.session_state[safe_key(column, val)] = True
            with c2:
                if st.button(f"Tout désélectionner", key=f"clear_all_{column}"):
                    for val in values:
                        st.session_state[safe_key(column, val)] = False

            st.markdown("")
            cols = st.columns(ncols)
            for idx, val in enumerate(values):
                key = safe_key(column, val)
                col_idx = idx % ncols
                with cols[col_idx]:
                    checked = st.session_state.get(key, True)
                    st.checkbox(str(val), key=key, value=checked)
        st.session_state["reset_filters"] = False
        return [val for val in values if st.session_state.get(safe_key(column, val), False)]

    region_filter = checkbox_grid_filter("Region", "Région")
    brand_filter = checkbox_grid_filter("VehBrand", "Marque du véhicule")
    gas_filter = checkbox_grid_filter("VehGas", "Type de carburant")
    area_filter = checkbox_grid_filter("Area", "Zone")

    # -----------------------------
    # Filtrage effectif
    # -----------------------------
    filtered_data = data[
        (data["VehAge"].between(vehage_min, vehage_max)) &
        (data["DrivAge"].between(drivage_min, drivage_max)) &
        (data["BonusMalus"].between(bonus_min, bonus_max)) &
        (data["Region"].astype(str).isin(region_filter)) &
        (data["VehBrand"].astype(str).isin(brand_filter)) &
        (data["VehGas"].astype(str).isin(gas_filter)) &
        (data["Area"].astype(str).isin(area_filter))
    ].copy()

    st.markdown(f"**Nombre de contrats sélectionnés : {filtered_data.shape[0]:,}**".replace(",", " "))
    # -----------------------------
    # KPIs
    # -----------------------------
    st.subheader("Chiffres clés")
    total_claims = filtered_data["ClaimAmount"].sum() if not filtered_data.empty else 0
    total_nb_claims = filtered_data["ClaimNb"].sum() if not filtered_data.empty else 0
    avg_freq = filtered_data["ClaimFrequency"].mean() if not filtered_data.empty else 0
    avg_sev = filtered_data.loc[filtered_data["ClaimNb"] > 0, "Severity"].mean() if not filtered_data.empty else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Nombre de contrats", f"{filtered_data.shape[0]:,}".replace(",", " "))
    col2.metric("Fréquence moyenne", f"{avg_freq:.4f}")
    col3.metric("Sévérité moyenne (€)", f"{avg_sev:,.0f}".replace(",", " "))
    col4.metric("Nombre total de sinistres", f"{int(total_nb_claims):,}".replace(",", " "))
    col5.metric("Montant total des sinistres (€)", f"{total_claims:,.0f}".replace(",", " "))


    # -----------------------------
    # Graphiques
    # -----------------------------
    st.subheader("Distribution des montants de sinistres")
    if not filtered_data.empty and (filtered_data["ClaimAmount"] > 0).any():
        fig, ax = plt.subplots()
        sns.histplot(
            data=filtered_data[filtered_data["ClaimAmount"] > 0],
            x="ClaimAmount",
            bins=60,
            log_scale=(True, False),
            kde=True,
            color=COLOR_PRIMARY,
            ax=ax
        )
        ax.set_xlabel("Montant du sinistre (€)")
        ax.set_ylabel("Nombre de sinistres")
        st.pyplot(fig)
    else:
        st.info("Aucune donnée à afficher pour les montants (vérifiez vos filtres).")

    st.subheader("Distribution du nombre de sinistres (ClaimNb)")
    if not filtered_data.empty:
        fig, ax = plt.subplots()
        sns.countplot(
            x="ClaimNb",
            data=filtered_data,
            order=sorted(filtered_data["ClaimNb"].unique()),
            color=COLOR_PRIMARY,
            ax=ax
        )
        ax.set_yscale("log")
        ax.set_xlabel("Nombre de sinistres")
        ax.set_ylabel("Nombre de contrats (log scale)")
        st.pyplot(fig)

    st.subheader("Fréquence moyenne par Région")
    if not filtered_data.empty:
        freq_region = filtered_data.groupby("Region")["ClaimFrequency"].mean().sort_values(ascending=False).reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=freq_region, x="Region", y="ClaimFrequency", color=COLOR_PRIMARY, ax=ax)
        ax.set_ylabel("Fréquence moyenne")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Sévérité moyenne par Région")
    if not filtered_data.empty and (filtered_data["ClaimNb"] > 0).any():
        sev_region = filtered_data[filtered_data["ClaimNb"] > 0].groupby("Region")["Severity"].mean().sort_values(ascending=False).reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=sev_region, x="Region", y="Severity", color=COLOR_PRIMARY, ax=ax)
        ax.set_ylabel("Sévérité moyenne (€)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# =============================================================================
# PAGE 3 : MODELISATION
# =============================================================================
elif menu == "Modélisation & Simulation":
    st.title("Modélisation & Estimation de la Prime Pure")
    st.markdown("Testez différents paramètres pour estimer la prime pure.")

    # -----------------------------
    # Création des classes
    # -----------------------------
    bins_age = [17, 20, 30, 40, 50, 60, 70, 80, 120]
    labels_age = ['18-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+']
    data['DrivAge_class'] = pd.cut(data['DrivAge'], bins=bins_age, labels=labels_age)

    bins_power = [0, 4, 6, 8, 10, 12, 15, 18, 21, 25, 30, 40, 60, 80, 100]
    labels_power = [
        '1-4 CV', '5-6 CV', '7-8 CV', '9-10 CV', '11-12 CV',
        '13-15 CV', '16-18 CV', '19-21 CV', '22-25 CV', '26-30 CV',
        '31-40 CV', '41-60 CV', '61-80 CV', '81+ CV'
    ]
    data['VehPower_class'] = pd.cut(data['VehPower'], bins=bins_power, labels=labels_power)

    @st.cache_resource
    def load_or_train_models(data):
        model_freq_path = "glm_freq.pkl"
        model_sev_path = "glm_sev.pkl"
        if os.path.exists(model_freq_path) and os.path.exists(model_sev_path):
            glm_freq = joblib.load(model_freq_path)
            glm_sev = joblib.load(model_sev_path)
            return glm_freq, glm_sev
        else:
            formula_freq = (
                "ClaimNb ~ VehPower_class + VehAge + DrivAge_class + BonusMalus + "
                "Area + VehBrand + VehGas + Density + Region"
            )
            glm_freq = smf.glm(
                formula=formula_freq,
                data=data,
                family=sm.families.Poisson(),
                offset=np.log(data["Exposure"])
            ).fit()

            data_sev = data[data["ClaimAmount"] > 0].copy()
            data_sev["Severity"] = data_sev["ClaimAmount"] / data_sev["ClaimNb"]
            formula_sev = (
                "Severity ~ VehPower_class + VehAge + DrivAge_class + BonusMalus + "
                "Area + VehBrand + VehGas + Density + Region"
            )
            glm_sev = smf.glm(
                formula=formula_sev,
                data=data_sev,
                family=sm.families.Gamma(sm.families.links.log())
            ).fit()

            joblib.dump(glm_freq, model_freq_path)
            joblib.dump(glm_sev, model_sev_path)
            return glm_freq, glm_sev

    glm_freq, glm_sev = load_or_train_models(data)
    st.success("Modèles prêts à l'emploi")

    # -----------------------------
    # Interface utilisateur
    # -----------------------------
    st.subheader("Saisissez les caractéristiques du contrat")
    col1, col2, col3 = st.columns(3)
    with col1:
        veh_age = st.slider("Âge du véhicule", 0, 20, 5)
        veh_power_class = st.selectbox("Puissance du véhicule", labels_power)
        veh_brand = st.selectbox("Marque du véhicule", sorted(data["VehBrand"].unique()))
    with col2:
        driv_age = st.slider("Âge du conducteur", 18, 90, 40)
        driv_age_class = pd.cut([driv_age], bins=bins_age, labels=labels_age)[0]
        bonus_malus = st.slider("Bonus-Malus", int(data["BonusMalus"].min()), int(data["BonusMalus"].max()), 100)
        veh_gas = st.selectbox("Type de carburant", sorted(data["VehGas"].unique()))
    with col3:
        area = st.selectbox("Zone", sorted(data["Area"].unique()))
        region = st.selectbox("Région", sorted(data["Region"].unique()))
        density = st.number_input("Densité (hab/km²)", 0, 50000, 3000)
        exposure = st.number_input("Durée d’exposition (années)", 0.0, 1.0, 1.0, 0.1)

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

    contract["ExpectedClaims"] = glm_freq.predict(contract)
    contract["ExpectedSeverity"] = glm_sev.predict(contract)
    contract["PurePremium"] = contract["ExpectedClaims"] * contract["ExpectedSeverity"]

    st.markdown("---")
    st.subheader("Résultats de la simulation")
    st.write(f"""
    - **Fréquence attendue (sinistres/an)** : {contract['ExpectedClaims'].iloc[0]:.4f}  
    - **Sévérité attendue (€ / sinistre)** : {contract['ExpectedSeverity'].iloc[0]:,.0f} €  
    - **Prime pure estimée (€ / an)** : **{contract['PurePremium'].iloc[0]:,.0f} €**
    """)
    st.progress(min(contract["PurePremium"].iloc[0] / 2000, 1.0))

