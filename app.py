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
st.title("🔬 CalculLAB – Application Scientifique en Ligne")

# Menu de navigation principal
menu = st.sidebar.selectbox("Choisissez un module", [
    "Accueil", 
    "Calcul d’intégrale", 
    "Interpolation",
    "Analyse de fonction de transfert",
    "Équations différentielles",
    "Intégration numérique",
    "Systèmes linéaires",
    "Décomposition LU",
    "Applications Laser",
    "Optimisation linéaire",
    "Séries de Fourier"
])

# Accueil
if menu == "Accueil":
    st.markdown("""
    ## Bienvenue sur CalculLAB Web
    
    Une plateforme de calcul scientifique complète destinée aux étudiants et chercheurs.
    
    **✨ Fonctionnalités disponibles:**
    - Calcul d'intégrales définies avec visualisation graphique
    - Interpolation linéaire de données
    - Analyse de fonctions de transfert (réponse temporelle, diagramme de Bode)
    - Résolution d'équations différentielles avec différentes méthodes numériques
    - Intégration numérique (méthodes des trapèzes, Simpson, point milieu)
    - Résolution de systèmes linéaires (Jacobi, Gauss-Seidel)
    - Décomposition LU de matrices
    - Applications laser (pertes par cavité, profil gaussien)
    - Optimisation linéaire avec Pulp
    - Analyse de séries de Fourier
    
    Choisissez un outil dans le menu à gauche pour commencer!
    """)
    
    st.image("https://images.unsplash.com/photo-1581091226033-d5c48150dbaa?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80", 
             caption="Laboratoire scientifique moderne")

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

# Module 3 : Analyse de fonction de transfert
elif menu == "Analyse de fonction de transfert":
    st.header("📊 Analyse de Fonction de Transfert")
    
    col1, col2 = st.columns(2)
    with col1:
        num_str = st.text_input("Numérateur (coefficients séparés par des virgules)", "1.0")
        num = [float(i.strip()) for i in num_str.split(',')]
    with col2:
        den_str = st.text_input("Dénominateur (coefficients séparés par des virgules)", "1.0, 1.0")
        den = [float(i.strip()) for i in den_str.split(',')]
    
    if st.button("Analyser la fonction de transfert"):
        try:
            sys_tf = TransferFunction(num, den)
            
            # Calcul des pôles et zéros
            zeros = sys_tf.zeros()
            poles = sys_tf.poles()
            order = len(den) - 1
            stable = all(np.real(p) < 0 for p in poles)
            
            # Affichage des résultats
            st.subheader("Propriétés du système")
            st.write(f"Ordre du système: {order}")
            st.write(f"Pôles: {poles}")
            st.write(f"Zéros: {zeros}")
            st.write(f"Stabilité: {'Stable' if stable else 'Instable'}")
            
            # Analyse pour les systèmes du premier et deuxième ordre
            if order == 1:
                try:
                    tau = -1 / np.real(poles[0])
                    st.write(f"Constante de temps τ: {tau:.4f} s")
                    st.write(f"Temps de réponse ≈ 4τ: {4 * tau:.4f} s")
                    st.write(f"Temps de montée ≈ 2.2τ: {2.2 * tau:.4f} s")
                except:
                    st.warning("Erreur dans l'analyse du premier ordre")
            
            elif order == 2:
                try:
                    wn = np.abs(poles[0])
                    zeta = -np.real(poles[0]) / wn if wn != 0 else 0
                    gain_statique = num[-1] / den[-1] if den[-1] != 0 else 0
                    
                    st.write(f"Gain statique: {gain_statique:.4f}")
                    st.write(f"Amortissement ζ: {zeta:.4f}")
                    st.write(f"Pulsation propre ω₀: {wn:.4f} rad/s")
                    
                    if zeta < 1:
                        Mp = np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2)) * 100
                        tr = 1.8 / wn
                        fc = wn * np.sqrt(1 - zeta ** 2)
                        ts = 4 / (zeta * wn)
                        
                        st.write(f"Dépassement Mp: {Mp:.2f}%")
                        st.write(f"Temps de montée ≈: {tr:.4f} s")
                        st.write(f"Fréquence de coupure ≈: {fc:.4f} rad/s")
                        st.write(f"Temps de stabilisation ≈: {ts:.4f} s")
                    else:
                        st.write("Système sur-amorti (ζ ≥ 1)")
                except Exception as e:
                    st.warning(f"Erreur dans l'analyse du deuxième ordre: {e}")
            
            # Réponse indicielle
            st.subheader("Réponse indicielle")
            t, y = step(sys_tf)
            fig1, ax1 = plt.subplots()
            ax1.plot(t, y)
            ax1.set_title("Réponse indicielle")
            ax1.set_xlabel("Temps (s)")
            ax1.set_ylabel("Amplitude")
            ax1.grid(True)
            st.pyplot(fig1)
            
            # Diagramme de Bode
            st.subheader("Diagramme de Bode")
            w, mag, phase = bode(sys_tf)
            
            fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(8, 8))
            ax2.semilogx(w, mag)
            ax2.set_title("Diagramme de Bode - Gain")
            ax2.set_ylabel("Gain (dB)")
            ax2.grid(True)
            
            ax3.semilogx(w, phase)
            ax3.set_title("Diagramme de Bode - Phase")
            ax3.set_xlabel("Fréquence (rad/s)")
            ax3.set_ylabel("Phase (°)")
            ax3.grid(True)
            
            st.pyplot(fig2)
            
        except Exception as e:
            st.error(f"Erreur lors de l'analyse: {e}")

# Module 4 : Équations différentielles
elif menu == "Équations différentielles":
    st.header("📉 Résolution d'Équations Différentielles")
    
    method = st.selectbox("Méthode numérique", 
                         ["Euler explicite", "Euler implicite", "Heun", "RK2", "RK4", "Crank-Nicolson"])
    
    col1, col2 = st.columns(2)
    with col1:
        eq_str = st.text_input("Équation (dy/dt = f(y, t))", "y - t")
        y0 = st.number_input("Condition initiale y(t0)", value=1.0)
    with col2:
        t0 = st.number_input("t initial (t0)", value=0.0)
        t1 = st.number_input("t final (t1)", value=10.0)
        n_points = st.number_input("Nombre de points", value=100, min_value=10)
    
    if st.button("Résoudre l'équation différentielle"):
        try:
            t = np.linspace(t0, t1, n_points)
            dt = t[1] - t[0]
            
            # Définition de la fonction
            def f(y, t):
                return eval(eq_str, {"y": y, "t": t, "np": np})
            
            # Résolution selon la méthode choisie
            if method == "Euler explicite":
                y = np.zeros(n_points)
                y[0] = y0
                for i in range(1, n_points):
                    y[i] = y[i-1] + f(y[i-1], t[i-1]) * dt
            
            elif method == "Euler implicite":
                y = np.zeros(n_points)
                y[0] = y0
                for i in range(1, n_points):
                    def g(ynext):
                        return ynext - y[i-1] - dt * f(ynext, t[i])
                    y[i] = fsolve(g, y[i-1])[0]
            
            elif method == "Heun":
                y = np.zeros(n_points)
                y[0] = y0
                for i in range(1, n_points):
                    y_pred = y[i-1] + dt * f(y[i-1], t[i-1])
                    y[i] = y[i-1] + dt/2 * (f(y[i-1], t[i-1]) + f(y_pred, t[i]))
            
            elif method == "RK2":
                y = np.zeros(n_points)
                y[0] = y0
                for i in range(1, n_points):
                    k1 = f(y[i-1], t[i-1])
                    k2 = f(y[i-1] + dt*k1/2, t[i-1] + dt/2)
                    y[i] = y[i-1] + dt * k2
            
            elif method == "RK4":
                y = np.zeros(n_points)
                y[0] = y0
                for i in range(1, n_points):
                    k1 = f(y[i-1], t[i-1])
                    k2 = f(y[i-1] + dt*k1/2, t[i-1] + dt/2)
                    k3 = f(y[i-1] + dt*k2/2, t[i-1] + dt/2)
                    k4 = f(y[i-1] + dt*k3, t[i-1] + dt)
                    y[i] = y[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            elif method == "Crank-Nicolson":
                y = np.zeros(n_points)
                y[0] = y0
                for i in range(1, n_points):
                    def g(ynext):
                        return ynext - y[i-1] - dt/2 * (f(y[i-1], t[i-1]) + f(ynext, t[i]))
                    y[i] = fsolve(g, y[i-1])[0]
            
            # Affichage des résultats
            st.subheader(f"Solution par méthode {method}")
            fig, ax = plt.subplots()
            ax.plot(t, y, 'b-', label='Solution numérique')
            ax.set_xlabel('t')
            ax.set_ylabel('y(t)')
            ax.set_title(f"Résolution par {method}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            # Tableau de résultats
            df = pd.DataFrame({
                'Temps': t[:10],
                'y(t)': y[:10]
            })
            st.write("Aperçu des résultats (10 premiers points):")
            st.dataframe(df)
            
        except Exception as e:
            st.error(f"Erreur lors de la résolution: {e}")

# Module 5 : Intégration numérique
elif menu == "Intégration numérique":
    st.header("🧮 Intégration Numérique")
    
    col1, col2 = st.columns(2)
    with col1:
        f_str = st.text_input("Fonction f(x)", "np.sin(x)")
        a = st.number_input("Borne inférieure a", value=0.0)
        b = st.number_input("Borne supérieure b", value=np.pi)
    with col2:
        n = st.number_input("Nombre de subdivisions", value=100, min_value=2)
        method = st.selectbox("Méthode d'intégration", 
                             ["Trapèzes", "Simpson", "Point milieu"])
    
    if st.button("Calculer l'intégrale"):
        try:
            f = eval(f"lambda x: {f_str}")
            
            # Calcul selon la méthode choisie
            if method == "Trapèzes":
                h = (b - a) / n
                x = np.linspace(a, b, n + 1)
                y = f(x)
                result = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
                
            elif method == "Simpson":
                if n % 2 != 0:
                    n += 1
                h = (b - a) / n
                x = np.linspace(a, b, n + 1)
                y = f(x)
                result = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
                
            elif method == "Point milieu":
                h = (b - a) / n
                x_mid = np.linspace(a + h/2, b - h/2, n)
                y_mid = f(x_mid)
                result = h * np.sum(y_mid)
            
            # Calcul de la valeur exacte
            exact_result, _ = quad(f, a, b)
            
            # Affichage des résultats
            st.success(f"Résultat ({method}): {result:.6f}")
            st.info(f"Valeur exacte: {exact_result:.6f}")
            st.write(f"Erreur absolue: {abs(result - exact_result):.6f}")
            st.write(f"Erreur relative: {abs((result - exact_result)/exact_result)*100:.4f}%")
            
            # Visualisation
            x_vals = np.linspace(a, b, 300)
            y_vals = f(x_vals)
            
            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals, 'b-', label=f'f(x) = {f_str}')
            
            if method == "Trapèzes":
                for i in range(n):
                    x_trap = [x[i], x[i], x[i+1], x[i+1]]
                    y_trap = [0, y[i], y[i+1], 0]
                    ax.fill(x_trap, y_trap, 'r', alpha=0.2)
                ax.set_title("Méthode des Trapèzes")
            
            elif method == "Simpson":
                for i in range(0, n, 2):
                    x_simp = [x[i], x[i], x[i+2], x[i+2]]
                    y_simp = [0, y[i], y[i+2], 0]
                    ax.fill(x_simp, y_simp, 'g', alpha=0.2)
                ax.set_title("Méthode de Simpson")
            
            elif method == "Point milieu":
                for i in range(n):
                    x_mid_i = a + (i + 0.5) * h
                    x_rect = [x_mid_i - h/2, x_mid_i - h/2, x_mid_i + h/2, x_mid_i + h/2]
                    y_rect = [0, f(x_mid_i), f(x_mid_i), 0]
                    ax.fill(x_rect, y_rect, 'purple', alpha=0.2)
                ax.set_title("Méthode du Point Milieu")
            
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Erreur lors du calcul: {e}")

# Module 6 : Systèmes linéaires
elif menu == "Systèmes linéaires":
    st.header("🧮 Résolution de Systèmes Linéaires")
    
    size = st.selectbox("Taille du système", [2, 3, 4, 5], index=1)
    st.write("Entrez la matrice A et le vecteur b:")
    
    # Création de la matrice A
    st.subheader("Matrice A")
    a_matrix = []
    cols = st.columns(size)
    for i in range(size):
        row = []
        for j in range(size):
            with cols[j]:
                row.append(st.number_input(f"A[{i+1},{j+1}]", value=1.0 if i == j else 0.0, key=f"a_{i}_{j}"))
        a_matrix.append(row)
    
    # Création du vecteur b
    st.subheader("Vecteur b")
    b_vector = []
    cols = st.columns(size)
    for i in range(size):
        with cols[i]:
            b_vector.append(st.number_input(f"b[{i+1}]", value=1.0, key=f"b_{i}"))
    
    method = st.selectbox("Méthode de résolution", 
                         ["Directe (numpy.linalg.solve)", 
                          "Jacobi (itératif)", 
                          "Gauss-Seidel (itératif)"])
    
    if method in ["Jacobi (itératif)", "Gauss-Seidel (itératif)"]:
        max_iter = st.number_input("Nombre maximal d'itérations", value=100, min_value=10)
        tolerance = st.number_input("Tolérance", value=1e-6, format="%e")
    
    if st.button("Résoudre le système"):
        try:
            A = np.array(a_matrix)
            b = np.array(b_vector)
            
            # Résolution selon la méthode choisie
            if method == "Directe (numpy.linalg.solve)":
                x = np.linalg.solve(A, b)
                st.success("Solution trouvée par méthode directe")
            
            elif method == "Jacobi (itératif)":
                x0 = np.zeros(size)
                x = x0.copy()
                for it in range(max_iter):
                    x_old = x.copy()
                    for i in range(size):
                        sigma = 0
                        for j in range(size):
                            if j != i:
                                sigma += A[i, j] * x_old[j]
                        x[i] = (b[i] - sigma) / A[i, i]
                    
                    # Vérification de la convergence
                    if np.linalg.norm(x - x_old) < tolerance:
                        st.success(f"Convergence atteinte en {it+1} itérations")
                        break
                else:
                    st.warning(f"Convergence non atteinte après {max_iter} itérations")
            
            elif method == "Gauss-Seidel (itératif)":
                x0 = np.zeros(size)
                x = x0.copy()
                for it in range(max_iter):
                    x_old = x.copy()
                    for i in range(size):
                        sigma1 = np.dot(A[i, :i], x[:i])
                        sigma2 = np.dot(A[i, i+1:], x_old[i+1:])
                        x[i] = (b[i] - sigma1 - sigma2) / A[i, i]
                    
                    # Vérification de la convergence
                    if np.linalg.norm(x - x_old) < tolerance:
                        st.success(f"Convergence atteinte en {it+1} itérations")
                        break
                else:
                    st.warning(f"Convergence non atteinte après {max_iter} itérations")
            
            # Affichage des résultats
            st.subheader("Solution du système")
            df = pd.DataFrame({
                'Variable': [f'x{i+1}' for i in range(size)],
                'Valeur': x
            })
            st.dataframe(df)
            
            # Vérification de la solution
            st.subheader("Vérification")
            residual = np.dot(A, x) - b
            st.write(f"Résidu (norme L2): {np.linalg.norm(residual):.6e}")
            
        except Exception as e:
            st.error(f"Erreur lors de la résolution: {e}")

# Module 7 : Décomposition LU
elif menu == "Décomposition LU":
    st.header("🔍 Décomposition LU")
    
    size = st.selectbox("Taille de la matrice", [2, 3, 4, 5], index=1)
    st.write("Entrez la matrice A:")
    
    # Création de la matrice A
    a_matrix = []
    cols = st.columns(size)
    for i in range(size):
        row = []
        for j in range(size):
            with cols[j]:
                row.append(st.number_input(f"A[{i+1},{j+1}]", value=1.0 if i == j else 0.0, key=f"a_{i}_{j}"))
        a_matrix.append(row)
    
    if st.button("Calculer la décomposition LU"):
        try:
            A = np.array(a_matrix, dtype=float)
            n = size
            
            # Initialisation des matrices L et U
            L = np.eye(n)
            U = np.zeros((n, n))
            
            # Décomposition LU
            for i in range(n):
                # Calcul de U
                for k in range(i, n):
                    s = 0
                    for j in range(i):
                        s += L[i, j] * U[j, k]
                    U[i, k] = A[i, k] - s
                
                # Calcul de L
                for k in range(i+1, n):
                    s = 0
                    for j in range(i):
                        s += L[k, j] * U[j, i]
                    L[k, i] = (A[k, i] - s) / U[i, i]
            
            # Vérification
            A_reconstructed = np.dot(L, U)
            
            # Affichage des résultats
            st.subheader("Matrice L (triangulaire inférieure)")
            st.write(L)
            
            st.subheader("Matrice U (triangulaire supérieure)")
            st.write(U)
            
            st.subheader("Vérification: L * U")
            st.write(A_reconstructed)
            
            st.subheader("Erreur de reconstruction")
            st.write(f"Norme de Frobenius: {np.linalg.norm(A - A_reconstructed):.6e}")
            
        except Exception as e:
            st.error(f"Erreur lors de la décomposition: {e}")

# Module 8 : Applications Laser
elif menu == "Applications Laser":
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

# Module 9 : Optimisation linéaire
elif menu == "Optimisation linéaire":
    st.header("📈 Optimisation Linéaire")
    
    # Sélection du type de problème
    problem_type = st.selectbox("Type de problème", ["Maximisation", "Minimisation"])
    
    # Définition des variables
    st.subheader("Variables de décision")
    num_vars = st.number_input("Nombre de variables", min_value=1, max_value=10, value=2)
    
    var_names = []
    obj_coeffs = []
    var_types = []
    
    cols = st.columns(3)
    cols[0].write("**Variable**")
    cols[1].write("**Coefficient objectif**")
    cols[2].write("**Type**")
    
    for i in range(num_vars):
        cols = st.columns(3)
        with cols[0]:
            var_names.append(st.text_input(f"Nom variable {i+1}", value=f"x{i+1}", key=f"var_name_{i}"))
        with cols[1]:
            obj_coeffs.append(st.number_input(f"Coef. obj. {i+1}", value=1.0, key=f"obj_coeff_{i}"))
        with cols[2]:
            var_types.append(st.selectbox(f"Type {i+1}", ["Libre", "≥ 0"], key=f"var_type_{i}"))
    
    # Définition des contraintes
    st.subheader("Contraintes")
    num_constraints = st.number_input("Nombre de contraintes", min_value=0, max_value=10, value=1)
    
    constraints = []
    for i in range(num_constraints):
        st.write(f"**Contrainte {i+1}**")
        cols = st.columns(4)
        with cols[0]:
            coeffs = []
            for j in range(num_vars):
                coeffs.append(st.number_input(f"Coef. x{j+1}", value=1.0, key=f"con_coeff_{i}_{j}"))
        with cols[1]:
            sign = st.selectbox("Signe", ["≤", "=", "≥"], key=f"con_sign_{i}")
        with cols[2]:
            rhs = st.number_input("Valeur", value=0.0, key=f"con_rhs_{i}")
        constraints.append((coeffs, sign, rhs))
    
    if st.button("Résoudre le problème"):
        try:
            # Création du problème
            if problem_type == "Maximisation":
                prob = pulp.LpProblem("Probleme_Optimisation", pulp.LpMaximize)
            else:
                prob = pulp.LpProblem("Probleme_Optimisation", pulp.LpMinimize)
            
            # Déclaration des variables
            variables = []
            for i in range(num_vars):
                if var_types[i] == "≥ 0":
                    var = pulp.LpVariable(var_names[i], lowBound=0)
                else:
                    var = pulp.LpVariable(var_names[i])
                variables.append(var)
            
            # Fonction objectif
            prob += pulp.lpSum([coeff * var for coeff, var in zip(obj_coeffs, variables)])
            
            # Contraintes
            for coeffs, sign, rhs in constraints:
                lhs = pulp.lpSum([coeff * var for coeff, var in zip(coeffs, variables)])
                if sign == "≤":
                    prob += lhs <= rhs
                elif sign == "≥":
                    prob += lhs >= rhs
                else:
                    prob += lhs == rhs
            
            # Résolution
            prob.solve()
            
            # Affichage des résultats
            st.subheader("Résultats")
            st.write(f"Statut de la solution: {pulp.LpStatus[prob.status]}")
            
            if prob.status == pulp.LpStatusOptimal:
                st.success(f"Valeur optimale de la fonction objectif: {pulp.value(prob.objective):.4f}")
                
                st.subheader("Valeurs optimales des variables")
                results = []
                for var in variables:
                    results.append({
                        "Variable": var.name,
                        "Valeur": var.varValue
                    })
                st.table(results)
                
                st.subheader("Valeurs duales (multiplicateurs de Lagrange)")
                dual_values = []
                for name, constraint in prob.constraints.items():
                    dual_values.append({
                        "Contrainte": name,
                        "Valeur duale": constraint.pi
                    })
                st.table(dual_values)
            else:
                st.warning("Aucune solution optimale trouvée")
                
        except Exception as e:
            st.error(f"Erreur lors de la résolution: {e}")

# Module 10 : Séries de Fourier
elif menu == "Séries de Fourier":
    st.header("📊 Analyse de Séries de Fourier")
    
    col1, col2 = st.columns(2)
    with col1:
        func_str = st.text_input("Fonction f(x)", "x**2 + sin(x)")
        L = st.text_input("Demi-période L", "pi")
    with col2:
        N = st.number_input("Nombre de termes N", min_value=1, value=5)
        x_min = st.number_input("x min", value=-3.0)
        x_max = st.number_input("x max", value=3.0)
    
    if st.button("Calculer la Série"):
        try:
            x = sp.symbols('x')
            f_expr = sp.sympify(func_str)
            L_val = sp.sympify(L)
            
            # Calcul des coefficients
            a0 = (1/(2*L_val)) * sp.integrate(f_expr, (x, -L_val, L_val))
            an = []
            bn = []
            for n in range(1, N+1):
                an.append((1/L_val) * sp.integrate(f_expr*sp.cos(n*sp.pi*x/L_val), (x, -L_val, L_val)))
                bn.append((1/L_val) * sp.integrate(f_expr*sp.sin(n*sp.pi*x/L_val), (x, -L_val, L_val)))
            
            # Construction de la série
            series = a0
            for n in range(1, N+1):
                series += an[n-1]*sp.cos(n*sp.pi*x/L_val) + bn[n-1]*sp.sin(n*sp.pi*x/L_val)
            
            # Affichage des coefficients
            st.subheader("Coefficients de Fourier")
            st.write(f"a₀ = {a0.evalf():.5f}")
            for i in range(N):
                st.write(f"a_{i+1} = {an[i].evalf():.5f}, b_{i+1} = {bn[i].evalf():.5f}")
            
            # Tracé
            x_vals = np.linspace(x_min, x_max, 500)
            f_lamb = sp.lambdify(x, f_expr, 'numpy')
            s_lamb = sp.lambdify(x, series, 'numpy')
            
            try:
                y_original = f_lamb(x_vals)
                y_approx = s_lamb(x_vals)
                
                fig, ax = plt.subplots()
                ax.plot(x_vals, y_original, 'b-', label='Fonction originale')
                ax.plot(x_vals, y_approx, 'r--', label=f'Approximation (N={N})')
                ax.set_xlabel('x')
                ax.set_ylabel('f(x)')
                ax.set_title("Série de Fourier")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"Erreur lors du tracé: {e}")
                st.write("Expression de la série:")
                st.latex(sp.latex(series.simplify()))
            
        except Exception as e:
            st.error(f"Erreur lors du calcul: {e}")
