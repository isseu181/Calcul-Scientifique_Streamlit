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
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    .back-button {
        background-color: #6c757d !important;
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

# Fonction pour afficher un bouton de retour
def show_back_button():
    if st.button("← Retour à l'accueil", key="back_button", 
                 use_container_width=True, 
                 type="secondary", 
                 help="Retourner à la page d'accueil"):
        st.session_state.current_page = "Accueil"
        st.experimental_rerun()

# Data Science
def data_science_page():
    st.header("📊 Outils Data Science")
    show_back_button()
    st.write("Cette section est en cours de développement...")
    
    # Simulation Data Science
    st.subheader("Simulation d'analyse de données")
    dataset_size = st.slider("Taille du jeu de données", 1000, 10000, 5000)
    noise_level = st.slider("Niveau de bruit", 0.1, 2.0, 0.5)
    
    # Génération de données
    np.random.seed(42)
    x = np.linspace(0, 10, dataset_size)
    y = np.sin(x) + noise_level * np.random.randn(dataset_size)
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(x, y, alpha=0.3, s=10, color="#3498db")
    ax.set_title("Visualisation de données simulées")
    ax.set_xlabel("Variable X")
    ax.set_ylabel("Variable Y")
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

# Gestion Énergétique
def energy_management_page():
    st.header("⚡ Gestion Énergétique")
    show_back_button()
    
    # Simulation énergétique
    st.subheader("Simulation de consommation énergétique")
    col1, col2 = st.columns(2)
    with col1:
        residential = st.slider("Consommation résidentielle", 100, 500, 200)
        industrial = st.slider("Consommation industrielle", 100, 800, 400)
    with col2:
        commercial = st.slider("Consommation commerciale", 50, 300, 150)
        renewable = st.slider("Part des énergies renouvelables (%)", 0, 100, 30)
    
    # Calculs
    total_consumption = residential + industrial + commercial
    renewable_energy = total_consumption * renewable / 100
    
    # Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Diagramme secteurs
    sectors = ['Résidentiel', 'Industriel', 'Commercial']
    consumption = [residential, industrial, commercial]
    ax1.pie(consumption, labels=sectors, autopct='%1.1f%%', startangle=90, 
            colors=['#3498db', '#2ecc71', '#e74c3c'])
    ax1.set_title("Répartition de la consommation")
    
    # Diagramme barres
    sources = ['Énergies fossiles', 'Énergies renouvelables']
    values = [total_consumption - renewable_energy, renewable_energy]
    ax2.bar(sources, values, color=['#f39c12', '#27ae60'])
    ax2.set_title("Sources d'énergie")
    ax2.set_ylabel("MWh")
    
    st.pyplot(fig)

# Navier-Stokes
def navier_stokes_page():
    st.header("🌊 Équations de Navier-Stokes")
    show_back_button()
    
    # Simulation d'écoulement fluide
    st.subheader("Simulation d'écoulement dans une conduite")
    col1, col2 = st.columns(2)
    with col1:
        viscosity = st.slider("Viscosité (Pa·s)", 0.001, 0.1, 0.01, step=0.001)
        velocity = st.slider("Vitesse d'écoulement (m/s)", 0.1, 10.0, 2.0)
    with col2:
        pipe_diameter = st.slider("Diamètre de la conduite (m)", 0.1, 2.0, 0.5)
        density = st.slider("Densité du fluide (kg/m³)", 500, 2000, 1000)
    
    # Calcul du nombre de Reynolds
    reynolds = density * velocity * pipe_diameter / viscosity
    
    # Détermination du régime d'écoulement
    if reynolds < 2000:
        flow_regime = "Laminaire"
        color = '#3498db'
    elif reynolds < 4000:
        flow_regime = "Transition"
        color = '#f39c12'
    else:
        flow_regime = "Turbulent"
        color = '#e74c3c'
    
    # Visualisation
    st.subheader(f"Régime d'écoulement: {flow_regime} (Re = {reynolds:.1f})")
    
    # Création d'un profil de vitesse
    x = np.linspace(-pipe_diameter/2, pipe_diameter/2, 100)
    if flow_regime == "Laminaire":
        y = velocity * (1 - (2*x/pipe_diameter)**2)
    else:
        y = velocity * (1 - abs(2*x/pipe_diameter)**(1/7))
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, color=color, linewidth=3)
    ax.fill_between(x, y, color=color, alpha=0.3)
    ax.set_title("Profil de vitesse dans la conduite")
    ax.set_xlabel("Position dans la conduite (m)")
    ax.set_ylabel("Vitesse (m/s)")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig)

# Numérisation
def numerization_page():
    st.header("🔢 Numérisation")
    show_back_button()
    
    # Simulation de numérisation
    st.subheader("Simulation de processus de numérisation")
    
    process_steps = [
        "Collecte de données",
        "Traitement initial",
        "Analyse",
        "Transformation numérique",
        "Automatisation"
    ]
    
    step_progress = {}
    for step in process_steps:
        step_progress[step] = st.slider(f"Progrès: {step}", 0, 100, 50)
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(process_steps))
    ax.barh(y_pos, [step_progress[step] for step in process_steps], color='#3498db')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(process_steps)
    ax.set_xlabel('Pourcentage de complétion')
    ax.set_title('Progression de la numérisation')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    st.pyplot(fig)

# Applications Laser
def laser_applications_page():
    st.header("🔦 Applications Laser")
    show_back_button()
    
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

# Gestion de la navigation
if st.session_state.current_page == "Accueil":
    show_home_page()

elif st.session_state.current_page == "Data Science":
    data_science_page()

elif st.session_state.current_page == "Gestion Énergétique":
    energy_management_page()

elif st.session_state.current_page == "Navier-Stokes":
    navier_stokes_page()

elif st.session_state.current_page == "Numérisation":
    numerization_page()

elif st.session_state.current_page == "Applications Laser":
    laser_applications_page()
