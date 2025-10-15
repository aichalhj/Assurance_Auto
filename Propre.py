#%% Imports et chargement des données
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Chargement des 2 datasets
freq = fetch_openml("freMTPL2freq", version=1, as_frame=True).frame
sev = fetch_openml("freMTPL2sev", version=1, as_frame=True).frame

# Jointure
data = freq.merge(sev.groupby("IDpol")["ClaimAmount"].sum().reset_index(),
                  on="IDpol", how="left")

# Remplacement des NaN (contrats sans sinistre)
data["ClaimAmount"] = data["ClaimAmount"].fillna(0)

#%% Création des classes
# Tranches d'âge du conducteur
bins_age = [17, 20, 30, 40, 50, 60, 70, 80, 120]
labels_age = ['18-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+']
data['DrivAge_class'] = pd.cut(data['DrivAge'], bins=bins_age, labels=labels_age, right=True)

# Classes de puissance du véhicule
data['VehPower_class'] = pd.qcut(data['VehPower'], q=3,
                                 labels=['Faible puissance', 'Moyenne puissance', 'Forte puissance'])

#%% GLM Poisson 
formula_freq = """
ClaimNb ~ VehPower_class + VehAge + DrivAge_class
+ BonusMalus + Area + VehBrand + VehGas + Density + Region
"""

#%%
import statsmodels.api as sm
import statsmodels.formula.api as smf
glm_freq = smf.glm(
    formula=formula_freq,
    data=data,
    family=sm.families.Poisson(),
    offset=np.log(data["Exposure"])
).fit()

print(glm_freq.summary())

# %%
#%% Prédiction
data['ExpectedClaims'] = glm_freq.predict(data)

print("Moyenne observée :", data["ClaimNb"].mean())
print("Moyenne prédite :", data["ExpectedClaims"].mean())

# %%
#Modélisation de la séverité


data_sev = data[data["ClaimAmount"] > 0].copy()

data_sev["Severity"] = data_sev["ClaimAmount"] / data_sev["ClaimNb"]

# %%
formula_sev =  """
Severity ~ VehPower_class + DrivAge_class
+ BonusMalus + Area + VehBrand + VehGas + Region
"""

# %%
glm_sev = smf.glm(
    formula=formula_sev,
    data=data_sev,
    family=sm.families.Gamma(sm.families.links.log())
).fit()

print(glm_sev.summary())

# %%
data["ExpectedSeverity"] = glm_sev.predict(data)

print("Moyenne observée :", data_sev["Severity"].mean())
print("Moyenne prédite :", data["ExpectedSeverity"].mean())
# %%
data["PurePremium"] = data["ExpectedClaims"] * data["ExpectedSeverity"]

# %%
data["PurePremium"].mean()
