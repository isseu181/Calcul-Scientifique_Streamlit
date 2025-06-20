import streamlit as st
import os

# Configuration de la page
st.set_page_config(
    page_title="Application de Calcul Scientifique",
    layout="centered",
    page_icon="🔬"
)

# Style CSS pour reproduire l'interface PyQt
st.markdown("""
<style>
    .main-container {
        max-width: 800px;
        padding-top: 2rem;
    }
    .title {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 30px;
        color: #2c3e50;
    }
    .button-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        padding: 12px 20px;
        font-size: 18px;
        border-radius: 8px;
        background-color: #3498db;
        color: white;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: scale(1.02);
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #eee;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

# Variable d'état pour gérer la page actuelle
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Accueil"

# Fonction pour afficher la page d'accueil
def show_home_page():
    st.markdown('<div class="title">Bienvenue dans l\'Application de Calcul Scientifique</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        
        if st.button("Outils Data Science"):
            st.session_state.current_page = "Data Science"
        
        if st.button("Gestion Énergétique"):
            st.session_state.current_page = "Gestion Énergétique"
        
        if st.button("Simulation: Laser"):
            st.session_state.current_page = "Simulation Laser"
        
        if st.button("Equations: Navier-Stokes"):
            st.session_state.current_page = "Équations Navier-Stokes"
        
        if st.button("Numérisation"):
            st.session_state.current_page = "Numérisation"
        
        if st.button("Quitter"):
            st.stop()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="footer">Application de Calcul Scientifique © 2023</div>', unsafe_allow_html=True)

# Pages pour les différentes fonctionnalités
def show_data_science_page():
    st.header("📊 Outils Data Science")
    st.write("Cette section est en cours de développement...")
    # Contenu Data Science ici

def show_energy_page():
    st.header("⚡ Gestion Énergétique")
    st.write("Cette section est en cours de développement...")
    # Contenu Gestion Énergétique ici

def show_laser_page():
    st.header("🔦 Simulation Laser")
    st.write("Cette section est en cours de développement...")
    # Contenu Simulation Laser ici

def show_navier_stokes_page():
    st.header("🌊 Équations de Navier-Stokes")
    st.write("Cette section est en cours de développement...")
    # Contenu Navier-Stokes ici

def show_numerization_page():
    st.header("🔢 Numérisation")
    st.write("Cette section est en cours de développement...")
    # Contenu Numérisation ici

# Gestion de la navigation
if st.session_state.current_page == "Accueil":
    show_home_page()

elif st.session_state.current_page == "Data Science":
    show_data_science_page()

elif st.session_state.current_page == "Gestion Énergétique":
    show_energy_page()

elif st.session_state.current_page == "Simulation Laser":
    show_laser_page()

elif st.session_state.current_page == "Équations Navier-Stokes":
    show_navier_stokes_page()

elif st.session_state.current_page == "Numérisation":
    show_numerization_page()
