
import streamlit as st
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from interface import (
    data_science_interface,
    energy_interface,
    laser_interface,
    navier_stokes_interface
)

st.set_page_config(page_title="CalculLAB Web", layout="centered")
st.title("🔬 CalculLAB – Application Scientifique en Ligne")

menu = st.sidebar.selectbox("Choisissez un module", ["Accueil", "Calcul d’intégrale", "Interpolation"])

# Accueil
if menu == "Accueil":
    st.markdown("Bienvenue sur **CalculLAB Web**, une plateforme de calcul scientifique destinée aux étudiants et lycéens.")
    st.markdown("Choisissez un outil dans le menu à gauche pour commencer.")

# Module 1 : Calcul d’intégrale
elif menu == "Calcul d’intégrale":
    st.header("🧮 Calcul d’intégrale définie")
    f_str = st.text_input("Entrer la fonction f(x)", "x**2")
    a = st.number_input("Borne inférieure (a)", value=0.0)
    b = st.number_input("Borne supérieure (b)", value=1.0)

    if st.button("Calculer l’intégrale"):
        try:
            f = eval("lambda x: " + f_str)
            result, _ = quad(f, a, b)
            st.success(f"Résultat de l'intégrale de {f_str} entre {a} et {b} : {result:.5f}")
            
            # Tracer la courbe
            x_vals = np.linspace(a, b, 300)
            y_vals = [f(x) for x in x_vals]
            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals, label=f"f(x) = {f_str}")
            ax.fill_between(x_vals, y_vals, alpha=0.3)
            ax.set_title("Graphique de la fonction")
            ax.grid(True)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur lors du calcul : {e}")

# Module 2 : Interpolation
elif menu == "Interpolation":
    st.header("📈 Interpolation linéaire")
    x_vals = st.text_input("Valeurs x (séparées par des virgules)", "0, 1, 2, 3")
    y_vals = st.text_input("Valeurs y (séparées par des virgules)", "1, 2, 1, 3")

    try:
        x = np.array([float(i.strip()) for i in x_vals.split(',')])
        y = np.array([float(i.strip()) for i in y_vals.split(',')])
        f_interp = interp1d(x, y, kind='linear')

        # Générer points interpolés
        x_new = np.linspace(np.min(x), np.max(x), 300)
        y_new = f_interp(x_new)

        # Afficher le graphique
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o', label='Points d’origine')
        ax.plot(x_new, y_new, '-', label='Interpolation linéaire')
        ax.set_title("Interpolation linéaire")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur : {e}")
