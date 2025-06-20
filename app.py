import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import TransferFunction, step, bode
from scipy.optimize import fsolve
import pulp
import sympy as sp
from sympy import symbols, integrate, pi, cos, sin, simplify, sympify, lambdify

st.set_page_config(page_title="CalculLAB Web", layout="centered", page_icon="🔬")

# Variable d'état pour gérer la page actuelle
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Accueil"

# Définir les pages disponibles
PAGES = {
    "Accueil": "Accueil",
    "Calcul d’intégrale": "Calcul d’intégrale",
    "Interpolation": "Interpolation",
    "Analyse de fonction de transfert": "Analyse de fonction de transfert",
    "Équations différentielles": "Équations différentielles",
    "Intégration numérique": "Intégration numérique",
    "Systèmes linéaires": "Systèmes linéaires",
    "Décomposition LU": "Décomposition LU",
    "Applications Laser": "Applications Laser",
    "Optimisation linéaire": "Optimisation linéaire",
    "Séries de Fourier": "Séries de Fourier",
    "Data Science": "Data Science",
    "Gestion Énergétique": "Gestion Énergétique",
    "Navier-Stokes": "Navier-Stokes",
    "Numérisation": "Numérisation"
}

# Page d'accueil
def show_home_page():
    st.markdown("""
    <style>
    .big-font {
        font-size:36px !important;
        text-align: center;
        margin-bottom: 30px;
    }
    .button-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
        max-width: 500px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        padding: 15px;
        font-size: 18px;
        border-radius: 10px;
        background-color: #4e73df;
        color: white;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2e59d9;
        transform: scale(1.02);
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font">Bienvenue dans l\'Application de Calcul Scientifique</p>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        
        if st.button("Outils Data Science"):
            st.session_state.current_page = "Data Science"
        
        if st.button("Gestion Énergétique"):
            st.session_state.current_page = "Gestion Énergétique"
        
        if st.button("Simulation: Laser"):
            st.session_state.current_page = "Applications Laser"
        
        if st.button("Equations: Navier-Stokes"):
            st.session_state.current_page = "Navier-Stokes"
        
        if st.button("Numérisation"):
            st.session_state.current_page = "Numérisation"
        
        if st.button("Quitter", key="quit"):
            st.stop()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="footer">🔬 CalculLAB - Plateforme Scientifique Complète</div>', unsafe_allow_html=True)

# Barre latérale pour la navigation
with st.sidebar:
    st.title("🔬 CalculLAB")
    selected_page = st.selectbox(
        "Navigation",
        list(PAGES.values()),
        index=list(PAGES.values()).index(st.session_state.current_page)
    )
    
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.experimental_rerun()

# Gestion des différentes pages
if st.session_state.current_page == "Accueil":
    show_home_page()

elif st.session_state.current_page == "Data Science":
    st.header("📊 Outils Data Science")
    st.write("Cette section est en cours de développement...")
    # Ajouter ici le contenu pour Data Science

elif st.session_state.current_page == "Gestion Énergétique":
    st.header("⚡ Gestion Énergétique")
    st.write("Cette section est en cours de développement...")
    # Ajouter ici le contenu pour la Gestion Énergétique

elif st.session_state.current_page == "Navier-Stokes":
    st.header("🌊 Équations de Navier-Stokes")
    st.write("Cette section est en cours de développement...")
    # Ajouter ici le contenu pour Navier-Stokes

elif st.session_state.current_page == "Numérisation":
    st.header("🔢 Numérisation")
    st.write("Cette section est en cours de développement...")
    # Ajouter ici le contenu pour la Numérisation

# Ajouter ici les autres pages (Calcul d'intégrale, Interpolation, etc.) 
# en copiant le code des modules précédents...
# (Le code des autres modules reste inchangé par rapport à la réponse précédente)

# Exemple pour le module Applications Laser
elif st.session_state.current_page == "Applications Laser":
    st.header("🔦 Applications Laser")
    
    app = st.selectbox("Sélectionnez l'application", ["Pertes par cavité", "Profil gaussien"])
    
    if app == "Pertes par cavité":
        st.subheader("Calcul des pertes par cavité laser")
        
        col1, col2 = st.columns(2)
        with col1:
            R1 = st.number_input("R1 (réflexion M1)", min_value=0.90, max_value=0.9999, value=0.99, step=0.0001, format="%.4f")
            R2 = st.number_input("R2 (réflexion M2)", min_value=0.90, max_value=0.9999, value=0.99, step=0.0001, format="%.4f")
        with col2:
            internal_loss = st.number_input("Pertes internes (fraction)", min_value=0.0, max_value=0.1, value=0.005, step=0.0001, format="%.4f")
        
        if st.button("Calculer les pertes"):
            T1 = 1 - R1
            T2 = 1 - R2
            total_loss = T1 + T2 + 2 * internal_loss
            
            st.subheader("Résultats")
            st.write(f"Transmission M1 (T1): {T1:.4f}")
            st.write(f"Transmission M2 (T2): {T2:.4f}")
            st.write(f"Pertes internes (par passage): {internal_loss:.4f}")
            st.success(f"**Pertes totales (par aller-retour): {total_loss*100:.2f}%**")
            
            # Visualisation
            labels = ['T1', 'T2', 'Pertes internes']
            sizes = [T1, T2, 2*internal_loss]
            
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title("Répartition des pertes")
            st.pyplot(fig)
    
    elif app == "Profil gaussien":
        st.subheader("Simulation du profil gaussien d'un faisceau laser")
        
        col1, col2 = st.columns(2)
        with col1:
            lam = st.number_input("λ (nm)", min_value=200.0, max_value=2000.0, value=532.0)
            w0 = st.number_input("w₀ (μm)", min_value=1.0, max_value=1000.0, value=50.0)
        with col2:
            z = st.number_input("z (mm)", min_value=-1000.0, max_value=1000.0, value=0.0)
            power = st.number_input("Puissance (mW)", min_value=0.01, value=10.0)
        
        if st.button("Afficher le profil gaussien"):
            lam_m = lam * 1e-9
            w0_m = w0 * 1e-6
            z_m = z * 1e-3
            power_w = power * 1e-3
            
            # Calcul des paramètres
            zR = np.pi * w0_m**2 / lam_m   # Rayon de Rayleigh
            wz = w0_m * np.sqrt(1 + (z_m / zR)**2)   # Taille du faisceau à z
            
            # Calcul de l'intensité
            r = np.linspace(-3*wz, 3*wz, 400)   # en mètres
            I0 = 2 * power_w / (np.pi * wz**2)
            I = I0 * np.exp(-2 * (r**2) / (wz**2))
            
            # Tracé
            fig, ax = plt.subplots()
            ax.plot(r * 1e6, I, color="#1976D2", lw=2)
            ax.set_title(f"Profil Gaussien à z = {z} mm")
            ax.set_xlabel("x (μm)")
            ax.set_ylabel("Intensité (W/m²)")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
            # Affichage des paramètres
            st.subheader("Paramètres du faisceau")
            st.write(f"Taille du faisceau à z (w(z)): {wz*1e6:.2f} μm")
            st.write(f"Rayon de Rayleigh (zR): {zR*1e3:.2f} mm")
            st.write(f"Intensité maximale (I0): {I0:.2e} W/m²")

# ... (Ajouter ici les autres modules comme dans la réponse précédente)
