import streamlit as st
import base64
from streamlit_extras.colored_header import colored_header

# Configuration de la page
st.set_page_config(
    page_title="Application de Calcul Scientifique",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé
st.markdown(f"""
    <style>
        body {{
            background: linear-gradient(135deg, #0c1b33, #1a365d);
            color: #e6f1ff;
        }}
        
        .stApp {{
            max-width: 900px;
            background: rgba(13, 26, 50, 0.85);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin: 2rem auto;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(86, 180, 239, 0.2);
        }}
        
        .title {{
            font-size: 2.8rem;
            text-align: center;
            background: linear-gradient(to right, #56b4ef, #2ecc71);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 1rem;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}
        
        .subtitle {{
            font-size: 1.2rem;
            color: #a3c7f7;
            text-align: center;
            margin-bottom: 2rem;
        }}
        
        .section {{
            margin: 2rem 0;
        }}
        
        .section-title {{
            font-size: 2rem;
            color: #56b4ef;
            position: relative;
            padding-left: 15px;
            margin-bottom: 1.5rem;
        }}
        
        .section-title:before {{
            content: '';
            position: absolute;
            left: 0;
            top: 10%;
            height: 80%;
            width: 5px;
            background: linear-gradient(to bottom, #56b4ef, #2ecc71);
            border-radius: 10px;
        }}
        
        .subsection {{
            margin: 1.5rem 0;
            padding-left: 25px;
        }}
        
        .subsection-title {{
            font-size: 1.6rem;
            color: #2ecc71;
            margin-bottom: 1rem;
        }}
        
        .tool-item {{
            padding: 15px 20px;
            margin-bottom: 15px;
            background: rgba(40, 65, 105, 0.6);
            border-radius: 12px;
            font-size: 1.1rem;
            border-left: 4px solid #56b4ef;
        }}
        
        .separator {{
            height: 2px;
            background: linear-gradient(to right, transparent, #56b4ef, transparent);
            margin: 2.5rem 0;
        }}
        
        .quit-btn {{
            display: block;
            width: 200px;
            margin: 2rem auto;
            padding: 15px;
            background: linear-gradient(to right, #e74c3c, #c0392b);
            color: white;
            text-align: center;
            text-decoration: none;
            border-radius: 50px;
            font-weight: bold;
            font-size: 1.1rem;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
        }}
        
        .quit-btn:hover {{
            transform: translateY(-3px);
            box-shadow: 0 7px 20px rgba(231, 76, 60, 0.5);
        }}
        
        @media (max-width: 768px) {{
            .title {{
                font-size: 2.2rem;
            }}
            
            .section-title {{
                font-size: 1.8rem;
            }}
        }}
    </style>
""", unsafe_allow_html=True)

# Animation atomique (SVG animé)
atom_svg = """
<svg width="150" height="150" viewBox="0 0 150 150" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto;">
    <circle cx="75" cy="75" r="10" fill="#2ecc71"/>
    
    <g transform="rotate(0,75,75)">
        <animateTransform attributeName="transform" type="rotate" from="0 75 75" to="360 75 75" dur="8s" repeatCount="indefinite"/>
        <circle cx="75" cy="75" r="30" stroke="#56b4ef" stroke-width="2" fill="none" stroke-dasharray="4 4" opacity="0.7"/>
        <circle cx="105" cy="75" r="5" fill="#56b4ef"/>
    </g>
    
    <g transform="rotate(120,75,75)">
        <animateTransform attributeName="transform" type="rotate" from="120 75 75" to="480 75 75" dur="12s" repeatCount="indefinite"/>
        <circle cx="75" cy="75" r="45" stroke="#56b4ef" stroke-width="2" fill="none" stroke-dasharray="4 4" opacity="0.7"/>
        <circle cx="120" cy="75" r="5" fill="#56b4ef"/>
    </g>
    
    <g transform="rotate(240,75,75)">
        <animateTransform attributeName="transform" type="rotate" from="240 75 75" to="600 75 75" dur="15s" repeatCount="indefinite"/>
        <circle cx="75" cy="75" r="60" stroke="#56b4ef" stroke-width="2" fill="none" stroke-dasharray="4 4" opacity="0.7"/>
        <circle cx="135" cy="75" r="5" fill="#56b4ef"/>
    </g>
</svg>
"""

# Fonction pour afficher l'animation atomique
def show_atom_animation():
    st.markdown(atom_svg, unsafe_allow_html=True)

# Contenu principal
st.markdown('<h1 class="title">Bienvenue dans l\'Application de Calcul Scientifique</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Plateforme avancée pour la modélisation, la simulation et l\'analyse de données scientifiques complexes</p>', unsafe_allow_html=True)

# Animation atomique
show_atom_animation()

# Section Outils Data Science
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Outils Data Science</h2>', unsafe_allow_html=True)

# Sous-section Gestion Énergétique
st.markdown('<div class="subsection">', unsafe_allow_html=True)
st.markdown('<h3 class="subsection-title">Gestion Énergétique</h3>', unsafe_allow_html=True)

# Liste des outils
tools = [
    "Simulation: Laser",
    "Equations: Navier-Stokes",
    "Numérisation"
]

for tool in tools:
    st.markdown(f'<div class="tool-item">➤ {tool}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Fin subsection
st.markdown('</div>', unsafe_allow_html=True)  # Fin section

# Séparateur
st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# Bouton Quitter
if st.button("Quitter", key="quit_button", use_container_width=True):
    st.success("Fermeture de l'application...")
    st.balloons()
    st.stop()

# Pied de page
st.markdown("""
    <div style="text-align: center; padding: 20px; font-size: 0.9rem; color: #7a9bc8; margin-top: 2rem;">
       
    </div>
""", unsafe_allow_html=True)
