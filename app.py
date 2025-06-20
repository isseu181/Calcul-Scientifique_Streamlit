import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
from scipy import integrate, interpolate
import sympy as sp

# Configuration de l'application
st.set_page_config(
    page_title="Révolution Numérique - Outils d'Aide à la Décision",
    layout="wide",
    page_icon="🔬"
)

# Style CSS personnalisé
st.markdown("""
<style>
    :root {
        --primary: #3498db;
        --secondary: #2c3e50;
        --accent: #e74c3c;
        --light: #ecf0f1;
        --dark: #2c3e50;
    }
    
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--secondary);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary);
    }
    
    .section-title {
        font-size: 1.8rem;
        color: var(--secondary);
        border-left: 5px solid var(--primary);
        padding-left: 1rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }
    
    .team-card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s;
    }
    
    .team-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .team-name {
        font-weight: bold;
        font-size: 1.2rem;
        color: var(--secondary);
    }
    
    .team-role {
        color: var(--primary);
        font-style: italic;
    }
    
    .card {
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .card-title {
        font-size: 1.4rem;
        color: var(--secondary);
        margin-bottom: 1rem;
    }
    
    .btn {
        background: var(--primary);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
        transition: background 0.3s;
        display: inline-block;
        text-align: center;
        margin-top: 1rem;
    }
    
    .btn:hover {
        background: #2980b9;
        text-decoration: none;
        color: white;
    }
    
    .objective-card {
        border-left: 4px solid var(--primary);
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        background: var(--light);
        color: var(--dark);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour la page d'accueil
def home_page():
    st.markdown('<div class="main-title">Révolution Numérique: Outils d\'Aide à la Décision</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; font-size: 1.2rem; color: #555;">
        Développement d'approches numériques effectives et intelligentes pour la résolution des problèmes de la société
    </div>
    """, unsafe_allow_html=True)
    
    # Objectifs du projet
    st.markdown('<div class="section-title">Objectifs Scientifiques</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card objective-card">
            <div class="card-title">OS1: Numérisation et dématérialisation</div>
            <p>Développement de technologies pour le marché de l'énergie, 
            visant à optimiser la gestion et la distribution des ressources énergétiques.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card objective-card">
            <div class="card-title">OS2: Sciences analytiques et données</div>
            <p>Émergence des sciences des données pour la prédiction et la personnalisation 
            ciblée des services dans divers domaines sociétaux.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card objective-card">
            <div class="card-title">OS3: Modélisation des fluides</div>
            <p>Résolution numérique des équations de Navier-Stokes en 2D appliquée 
            à un fluide réel circulant dans une conduite.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card objective-card">
            <div class="card-title">OS4: Applications laser</div>
            <p>Simulation des pertes par cavité laser et du profil gaussien de l'intensité du laser. 
            Conception d'une interface de calcul scientifique pour la communauté universitaire et secondaire.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Équipe de recherche
    st.markdown('<div class="section-title">Équipe de Recherche</div>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    team = [
        {"name": "Dr. MSDiallo", "role": "Responsable Scientifique"},
        {"name": "Adama GUEYE", "role": "Responsable Technique"},
        {"name": "Ahmadou Koumé SENE", "role": "Chercheur"},
        {"name": "Ndèye SARR", "role": "Chercheuse"},
        {"name": "Seydina Mouhamed Lybass POUYE", "role": "Chercheur"},
        {"name": "Alpha BA", "role": "Chercheur"}
    ]
    
    for i, member in enumerate(team):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="team-card">
                <div class="team-name">{member['name']}</div>
                <div class="team-role">{member['role']}</div>
            </div>
            """, unsafe_allow_html=True)

# Fonction pour l'OS1: Numérisation énergétique
def os1_page():
    st.markdown('<div class="section-title">OS1: Numérisation et Dématérialisation des Technologies du Marché de l\'Énergie</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p>Cette section développe des outils pour optimiser la gestion et la distribution 
        des ressources énergétiques grâce à la numérisation et à la dématérialisation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Simulation du réseau énergétique")
        st.markdown("""
        <div class="card">
            <p>Modélisation des flux énergétiques dans un réseau intelligent:</p>
            <ul>
                <li>Optimisation de la distribution</li>
                <li>Prévision de la demande</li>
                <li>Gestion des ressources renouvelables</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Graphique de prévision de la demande
        days = np.arange(1, 31)
        energy_demand = 500 + 100 * np.sin(days * 0.2) + np.random.normal(0, 20, 30)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(days, energy_demand, 'o-', color='#3498db')
        ax.set_title('Prévision de la demande énergétique (30 jours)')
        ax.set_xlabel('Jours')
        ax.set_ylabel('Demande (MW)')
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Optimisation des ressources")
        st.markdown("""
        <div class="card">
            <p>Outils d'aide à la décision pour la répartition des ressources:</p>
            <ul>
                <li>Algorithmes d'optimisation</li>
                <li>Analyse coût-bénéfice</li>
                <li>Impact environnemental</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Graphique de répartition des ressources
        sources = ['Solaire', 'Éolien', 'Hydraulique', 'Thermique']
        capacities = [120, 180, 220, 350]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(capacities, labels=sources, autopct='%1.1f%%', 
               colors=['#f1c40f', '#2ecc71', '#3498db', '#e74c3c'], 
               startangle=90)
        ax.set_title('Répartition des capacités énergétiques')
        st.pyplot(fig)

# Fonction pour l'OS2: Sciences des données
def os2_page():
    st.markdown('<div class="section-title">OS2: Sciences Analytiques et des Données</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p>Cette section développe des modèles prédictifs et des outils d'analyse de données 
        pour une personnalisation ciblée des services.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Analyse prédictive")
        st.markdown("""
        <div class="card">
            <p>Modèles de prédiction pour divers domaines:</p>
            <ul>
                <li>Prévision économique</li>
                <li>Analyse de marché</li>
                <li>Comportement des consommateurs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulation de données
        st.subheader("Simulation de modèle prédictif")
        n_points = st.slider("Nombre de points de données", 50, 500, 200)
        
        # Générer des données simulées
        np.random.seed(42)
        X = np.linspace(0, 10, n_points)
        y = 2 * X + 1 + np.random.normal(0, 2, n_points)
        
        # Régression linéaire
        coeffs = np.polyfit(X, y, 1)
        regression_line = np.poly1d(coeffs)
        
        # Affichage
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(X, y, alpha=0.6, color='#3498db', label='Données')
        ax.plot(X, regression_line(X), color='#e74c3c', linewidth=2, label='Modèle prédictif')
        ax.set_title('Modèle de régression linéaire')
        ax.set_xlabel('Variable indépendante')
        ax.set_ylabel('Variable dépendante')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Personnalisation des services")
        st.markdown("""
        <div class="card">
            <p>Outils pour une personnalisation ciblée:</p>
            <ul>
                <li>Segmentation de clientèle</li>
                <li>Recommandations personnalisées</li>
                <li>Optimisation de l'expérience utilisateur</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Clustering
        st.subheader("Segmentation de clientèle")
        
        # Génération de données de clustering
        np.random.seed(1)
        data1 = np.random.normal([2, 2], [0.5, 0.5], (100, 2))
        data2 = np.random.normal([5, 5], [0.5, 0.5], (100, 2))
        data3 = np.random.normal([8, 2], [0.5, 0.5], (100, 2))
        data = np.vstack([data1, data2, data3])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(data[:, 0], data[:, 1], alpha=0.6, color='#3498db')
        ax.set_title('Segmentation de clientèle (Clustering)')
        ax.set_xlabel('Caractéristique 1')
        ax.set_ylabel('Caractéristique 2')
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)

# Fonction pour l'OS3: Navier-Stokes
def os3_page():
    st.markdown('<div class="section-title">OS3: Résolution Numérique des Équations de Navier-Stokes</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p>Modélisation numérique d'un fluide réel circulant dans une conduite 
        en utilisant les équations de Navier-Stokes en 2D.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Paramètres de simulation")
        st.markdown("""
        <div class="card">
            <p>Configuration du modèle:</p>
        </div>
        """, unsafe_allow_html=True)
        
        viscosity = st.slider("Viscosité (ν)", 0.01, 0.1, 0.03)
        density = st.slider("Densité (ρ)", 0.5, 2.0, 1.0)
        pressure_grad = st.slider("Gradient de pression", 0.1, 2.0, 1.0)
        pipe_length = st.slider("Longueur de la conduite (m)", 5.0, 20.0, 10.0)
        pipe_diameter = st.slider("Diamètre de la conduite (m)", 0.5, 2.0, 1.0)
        
        # Calcul du nombre de Reynolds
        velocity = 1.0  # Vitesse moyenne
        reynolds = (density * velocity * pipe_diameter) / viscosity
        st.info(f"Nombre de Reynolds: {reynolds:.2f}")
        
        # Interprétation du nombre de Reynolds
        if reynolds < 2000:
            flow_regime = "Écoulement laminaire"
        elif reynolds > 4000:
            flow_regime = "Écoulement turbulent"
        else:
            flow_regime = "Écoulement de transition"
            
        st.success(f"Régime d'écoulement: {flow_regime}")
    
    with col2:
        st.subheader("Visualisation de l'écoulement")
        
        # Création d'une grille
        n = 50
        x = np.linspace(0, pipe_length, n)
        y = np.linspace(-pipe_diameter/2, pipe_diameter/2, n)
        X, Y = np.meshgrid(x, y)
        
        # Calcul de la vitesse (solution simplifiée)
        U = np.zeros_like(X)
        V = np.zeros_like(X)
        
        # Profil parabolique pour écoulement laminaire
        for i in range(n):
            for j in range(n):
                # Écoulement de Poiseuille (solution exacte)
                r = abs(Y[i, j])
                R = pipe_diameter/2
                U[i, j] = (pressure_grad/(4*viscosity)) * (R**2 - r**2)
        
        # Création de la visualisation
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Tracé du champ de vitesse
        speed = np.sqrt(U**2 + V**2)
        strm = ax.streamplot(X, Y, U, V, color=speed, cmap='viridis', density=1.5)
        fig.colorbar(strm.lines, ax=ax, label='Vitesse (m/s)')
        
        # Configuration du graphique
        ax.set_title('Écoulement dans une conduite')
        ax.set_xlabel('Longueur (m)')
        ax.set_ylabel('Hauteur (m)')
        ax.set_aspect('equal')
        
        st.pyplot(fig)

# Fonction pour l'OS4: Applications Laser
def os4_page():
    st.markdown('<div class="section-title">OS4: Applications Laser</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Pertes par cavité laser", "Profil gaussien"])
    
    with tab1:
        st.subheader("Simulation des pertes par cavité laser")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <p>Calcul des pertes dans une cavité laser:</p>
                <ul>
                    <li>Réflexions des miroirs</li>
                    <li>Pertes internes</li>
                    <li>Pertes totales par aller-retour</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            R1 = st.slider("Réflectivité Miroir 1 (R₁)", 0.90, 0.9999, 0.99, 0.0001)
            R2 = st.slider("Réflectivité Miroir 2 (R₂)", 0.90, 0.9999, 0.99, 0.0001)
            internal_loss = st.slider("Pertes internes (par passage)", 0.001, 0.05, 0.005, 0.0001)
            
            # Calcul des pertes
            T1 = 1 - R1
            T2 = 1 - R2
            total_loss = T1 + T2 + 2 * internal_loss
            
            st.success(f"Pertes totales par aller-retour: {total_loss*100:.2f}%")
            
            # Visualisation
            labels = ['Transmission M1', 'Transmission M2', 'Pertes internes']
            sizes = [T1, T2, 2*internal_loss]
            colors = ['#3498db', '#2ecc71', '#e74c3c']
            
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title("Répartition des pertes")
            st.pyplot(fig)
        
        with col2:
            # Diagramme de la cavité laser
            st.subheader("Schéma de la cavité laser")
            
            # Création d'un diagramme simple
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Dessiner la cavité
            ax.plot([0, 10], [0, 0], 'k-', linewidth=2)
            
            # Miroirs
            ax.plot([0, 0], [-0.5, 0.5], 'b-', linewidth=4)
            ax.plot([10, 10], [-0.5, 0.5], 'b-', linewidth=4)
            
            # Étiquettes
            ax.text(-0.5, 0, 'Miroir 1', fontsize=12, ha='right')
            ax.text(10.5, 0, 'Miroir 2', fontsize=12)
            
            # Rayon laser
            x = np.linspace(0, 10, 100)
            y = 0.1 * np.sin(2 * np.pi * x / 2)
            ax.plot(x, y, 'r-', linewidth=2)
            
            # Configuration
            ax.set_xlim(-1, 11)
            ax.set_ylim(-1, 1)
            ax.axis('off')
            ax.set_title("Cavité laser")
            
            st.pyplot(fig)
    
    with tab2:
        st.subheader("Profil Gaussien de l'Intensité Laser")
        
        col1, col2 = st.columns(2)
        
        with col1:
            wavelength = st.slider("Longueur d'onde (nm)", 200.0, 2000.0, 532.0)
            waist = st.slider("Taille du waist (μm)", 1.0, 100.0, 50.0)
            distance = st.slider("Distance (mm)", -100.0, 100.0, 0.0)
            power = st.slider("Puissance (mW)", 0.1, 100.0, 10.0)
            
            # Calculs
            wavelength_m = wavelength * 1e-9
            waist_m = waist * 1e-6
            distance_m = distance * 1e-3
            power_w = power * 1e-3
            
            # Paramètres du faisceau
            zR = np.pi * waist_m**2 / wavelength_m   # Rayon de Rayleigh
            wz = waist_m * np.sqrt(1 + (distance_m / zR)**2)  # Taille du faisceau
            I0 = 2 * power_w / (np.pi * wz**2)  # Intensité maximale
            
            st.info(f"Rayon de Rayleigh: {zR*1e3:.2f} mm")
            st.info(f"Taille du faisceau à z = {distance} mm: {wz*1e6:.2f} μm")
            st.info(f"Intensité maximale: {I0:.2e} W/m²")
        
        with col2:
            # Profil gaussien
            r = np.linspace(-3*wz, 3*wz, 200)
            I = I0 * np.exp(-2 * (r**2) / (wz**2))
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(r * 1e6, I, color='#e74c3c', linewidth=2)
            ax.set_title("Profil Gaussien de l'Intensité Laser")
            ax.set_xlabel("Distance radiale (μm)")
            ax.set_ylabel("Intensité (W/m²)")
            ax.grid(True, linestyle='--', alpha=0.3)
            
            st.pyplot(fig)

# Fonction pour la page d'interface de calcul
def calcul_scientifique_page():
    st.markdown('<div class="section-title">Interface de Calcul Scientifique</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p>Plateforme de calcul scientifique pour la communauté universitaire et secondaire, 
        intégrant divers outils numériques pour l'enseignement et la recherche.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Intégration numérique", "Équations différentielles", "Interpolation"])
    
    with tab1:
        st.subheader("Calcul numérique d'intégrales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            function = st.text_input("Fonction f(x)", "sin(x)")
            a = st.number_input("Borne inférieure a", value=0.0)
            b = st.number_input("Borne supérieure b", value=np.pi)
            method = st.selectbox("Méthode", ["Trapèzes", "Simpson", "Monte Carlo"])
            
            if st.button("Calculer l'intégrale"):
                try:
                    # Définition de la fonction
                    f = lambda x: eval(function, {"__builtins__": None}, 
                                       {"x": x, "sin": np.sin, "cos": np.cos, 
                                        "exp": np.exp, "log": np.log, "sqrt": np.sqrt})
                    
                    # Calcul selon la méthode
                    if method == "Trapèzes":
                        n = 100
                        x = np.linspace(a, b, n+1)
                        y = f(x)
                        result = (b-a)/n * (np.sum(y) - 0.5*(y[0] + y[-1]))
                    elif method == "Simpson":
                        n = 100
                        h = (b-a)/n
                        x = np.linspace(a, b, n+1)
                        y = f(x)
                        result = h/3 * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]))
                    else:  # Monte Carlo
                        n = 10000
                        x_rand = np.random.uniform(a, b, n)
                        y_rand = f(x_rand)
                        result = (b-a) * np.mean(y_rand)
                    
                    st.success(f"Résultat de l'intégrale: {result:.6f}")
                    
                    # Visualisation
                    x_vals = np.linspace(a, b, 200)
                    y_vals = f(x_vals)
                    
                    fig, ax = plt.subplots()
                    ax.plot(x_vals, y_vals, 'b-', linewidth=2)
                    ax.fill_between(x_vals, y_vals, alpha=0.3)
                    ax.set_title(f"Intégrale de {function}")
                    ax.set_xlabel("x")
                    ax.set_ylabel("f(x)")
                    ax.grid(True, linestyle='--', alpha=0.3)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
    
    with tab2:
        st.subheader("Résolution d'équations différentielles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            equation = st.text_input("Équation dy/dx", "y - x")
            y0 = st.number_input("Condition initiale y(0)", value=1.0)
            x0 = st.number_input("x initial", value=0.0)
            x1 = st.number_input("x final", value=5.0)
            n_points = st.slider("Nombre de points", 10, 500, 100)
            
            if st.button("Résoudre"):
                try:
                    # Définition de l'équation
                    def f(x, y):
                        return eval(equation, {"x": x, "y": y})
                    
                    # Résolution
                    x_vals = np.linspace(x0, x1, n_points)
                    y_vals = [y0]
                    
                    # Méthode d'Euler
                    h = (x1 - x0) / (n_points - 1)
                    for i in range(1, n_points):
                        y_next = y_vals[-1] + h * f(x_vals[i-1], y_vals[-1])
                        y_vals.append(y_next)
                    
                    # Visualisation
                    fig, ax = plt.subplots()
                    ax.plot(x_vals, y_vals, 'r-', linewidth=2)
                    ax.set_title(f"Solution de dy/dx = {equation}")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.grid(True, linestyle='--', alpha=0.3)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
    
    with tab3:
        st.subheader("Interpolation de données")
        
        st.markdown("""
        <div class="card">
            <p>Entrez vos données sous forme de listes séparées par des virgules</p>
        </div>
        """, unsafe_allow_html=True)
        
        x_input = st.text_input("Valeurs x", "1, 2, 3, 4, 5")
        y_input = st.text_input("Valeurs y", "1, 4, 2, 5, 3")
        
        if st.button("Interpoler"):
            try:
                # Conversion des données
                x = np.array([float(val.strip()) for val in x_input.split(',')])
                y = np.array([float(val.strip()) for val in y_input.split(',')])
                
                # Interpolation
                f = interpolate.interp1d(x, y, kind='cubic')
                x_new = np.linspace(min(x), max(x), 100)
                y_new = f(x_new)
                
                # Visualisation
                fig, ax = plt.subplots()
                ax.plot(x, y, 'bo', markersize=8, label='Données')
                ax.plot(x_new, y_new, 'r-', linewidth=2, label='Interpolation')
                ax.set_title("Interpolation des données")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.3)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Erreur: {str(e)}")

# Navigation principale
pages = {
    "Accueil": home_page,
    "OS1: Numérisation Énergétique": os1_page,
    "OS2: Sciences des Données": os2_page,
    "OS3: Navier-Stokes": os3_page,
    "OS4: Applications Laser": os4_page,
    "Interface de Calcul": calcul_scientifique_page
}

# Barre latérale de navigation
st.sidebar.title("🔬 Navigation")
selection = st.sidebar.radio("Sections", list(pages.keys()))

# Affichage de la page sélectionnée
pages[selection]()

# Pied de page
st.markdown("""
<div class="footer">
    Projet de Recherche - Révolution Numérique comme Outil d'Aide à la Décision<br>
    Université Cheikh Anta Diop de Dakar - 2023
</div>
""", unsafe_allow_html=True)
