import os
import sys
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QSize
from PyQt5.QtGui import QFont, QPixmap, QColor, QLinearGradient, QPainter, QBrush, QIcon
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
    QWidget, QGraphicsDropShadowEffect, QSpacerItem, QSizePolicy, 
    QFrame, QListWidget, QListWidgetItem
)
from PyQt5 import uic

# Style CSS pour l'application
APP_STYLE = """
QMainWindow {
    background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, 
                stop:0 #0c1b33, stop:1 #1a365d);
}

/* Boutons principaux */
.main-button {
    background-color: rgba(40, 65, 105, 0.6);
    color: #e6f1ff;
    border: none;
    border-radius: 12px;
    padding: 15px 20px;
    font-size: 16px;
    font-weight: bold;
    text-align: left;
    border-left: 4px solid #56b4ef;
    min-height: 60px;
}

.main-button:hover {
    background-color: rgba(50, 85, 135, 0.8);
    transform: translateX(10px);
}

.main-button:pressed {
    background-color: rgba(30, 55, 95, 0.8);
}

/* Bouton Quitter */
.quit-button {
    background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, 
                stop:0 #e74c3c, stop:1 #c0392b);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 12px 30px;
    font-size: 16px;
    font-weight: bold;
    min-width: 180px;
}

.quit-button:hover {
    background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, 
                stop:0 #ff6b6b, stop:1 #ff5252);
}

.quit-button:pressed {
    background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, 
                stop:0 #c0392b, stop:1 #e74c3c);
}

/* Titres */
.title-label {
    font-size: 36px;
    font-weight: bold;
    color: #56b4ef;
    background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, 
                stop:0 #56b4ef, stop:1 #2ecc71);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.subtitle-label {
    font-size: 18px;
    color: #a3c7f7;
    margin-bottom: 20px;
}

.section-title {
    font-size: 24px;
    color: #56b4ef;
    font-weight: bold;
    margin-bottom: 15px;
    padding-left: 10px;
    border-left: 4px solid qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, 
                stop:0 #56b4ef, stop:1 #2ecc71);
}

/* Conteneurs */
.main-container {
    background-color: rgba(13, 26, 50, 0.85);
    border-radius: 20px;
    padding: 30px;
    border: 1px solid rgba(86, 180, 239, 0.2);
}

.separator {
    background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, 
                stop:0 transparent, stop:0.5 #56b4ef, stop:1 transparent);
    height: 2px;
    margin: 25px 0;
}
"""

class AtomAnimation(QLabel):
    """Widget personnalisé pour l'animation atomique"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(150, 150)
        self.angle = 0
        self.timer_id = self.startTimer(50)  # Mise à jour toutes les 50ms
        
    def timerEvent(self, event):
        self.angle = (self.angle + 2) % 360
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Dessiner le noyau
        painter.setBrush(QBrush(QColor(46, 204, 113)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(65, 65, 20, 20)
        
        # Dessiner les orbites
        painter.setPen(QPen(QColor(86, 180, 239, 150), 2))
        painter.setBrush(Qt.NoBrush)
        
        # Orbite 1
        painter.drawEllipse(35, 35, 80, 80)
        painter.setBrush(QBrush(QColor(86, 180, 239)))
        x1 = 75 + 40 * (1 - abs(1 - 2 * abs(self.angle % 360 - 0) / 180))
        y1 = 75 + 40 * (1 - abs(1 - 2 * abs((self.angle + 90) % 360 - 0) / 180))
        painter.drawEllipse(int(x1 - 5), int(y1 - 5), 10, 10)
        
        # Orbite 2
        painter.drawEllipse(25, 25, 100, 100)
        x2 = 75 + 50 * (1 - abs(1 - 2 * abs((self.angle + 120) % 360 - 0) / 180))
        y2 = 75 + 50 * (1 - abs(1 - 2 * abs((self.angle + 210) % 360 - 0) / 180))
        painter.drawEllipse(int(x2 - 5), int(y2 - 5), 10, 10)
        
        # Orbite 3
        painter.drawEllipse(15, 15, 120, 120)
        x3 = 75 + 60 * (1 - abs(1 - 2 * abs((self.angle + 240) % 360 - 0) / 180))
        y3 = 75 + 60 * (1 - abs(1 - 2 * abs((self.angle + 330) % 360 - 0) / 180))
        painter.drawEllipse(int(x3 - 5), int(y3 - 5), 10, 10)
        
        painter.end()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Application de Calcul Scientifique")
        self.setMinimumSize(900, 700)
        self.setStyleSheet(APP_STYLE)
        
        # Widget central et layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Conteneur principal
        container = QFrame()
        container.setObjectName("main-container")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(40, 40, 40, 40)
        container_layout.setSpacing(20)
        
        # Titre principal
        title_label = QLabel("Bienvenue dans l'Application de Calcul Scientifique")
        title_label.setObjectName("title-label")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
        
        # Sous-titre
        subtitle_label = QLabel("Plateforme avancée pour la modélisation, la simulation et l'analyse de données scientifiques complexes")
        subtitle_label.setObjectName("subtitle-label")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setFont(QFont("Segoe UI", 12))
        subtitle_label.setWordWrap(True)
        
        # Animation atomique
        atom_animation = AtomAnimation()
        atom_animation.setAlignment(Qt.AlignCenter)
        
        # Section Outils Data Science
        section_title = QLabel("Outils Data Science")
        section_title.setObjectName("section-title")
        section_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        
        # Boutons d'outils
        tools_layout = QVBoxLayout()
        tools_layout.setSpacing(15)
        
        # Création des boutons
        self.data_science_button = self.create_tool_button("Data Science", "📊")
        self.energy_button = self.create_tool_button("Gestion Énergétique", "⚡")
        self.laser_button = self.create_tool_button("Simulation Laser", "🔦")
        self.navier_stokes_button = self.create_tool_button("Équations Navier-Stokes", "🌊")
        self.numerisation_button = self.create_tool_button("Numérisation", "🔢")
        
        # Ajout des boutons au layout
        tools_layout.addWidget(self.data_science_button)
        tools_layout.addWidget(self.energy_button)
        tools_layout.addWidget(self.laser_button)
        tools_layout.addWidget(self.navier_stokes_button)
        tools_layout.addWidget(self.numerisation_button)
        
        # Séparateur
        separator = QFrame()
        separator.setObjectName("separator")
        separator.setFixedHeight(2)
        
        # Bouton Quitter
        quit_button = QPushButton("Quitter l'Application")
        quit_button.setObjectName("quit-button")
        quit_button.setFont(QFont("Segoe UI", 12, QFont.Bold))
        quit_button.setCursor(Qt.PointingHandCursor)
        quit_button.clicked.connect(self.close)
        
        # Ajout d'un effet d'ombre
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 5)
        quit_button.setGraphicsEffect(shadow)
        
        # Ajout des widgets au conteneur
        container_layout.addWidget(title_label)
        container_layout.addWidget(subtitle_label)
        container_layout.addWidget(atom_animation, 0, Qt.AlignCenter)
        container_layout.addSpacing(20)
        container_layout.addWidget(section_title)
        container_layout.addLayout(tools_layout)
        container_layout.addWidget(separator)
        container_layout.addWidget(quit_button, 0, Qt.AlignCenter)
        
        # Ajout du conteneur au layout principal
        main_layout.addWidget(container)
        
        # Connexion des signaux
        self.connect_buttons()
        
        # Initialisation des fenêtres filles (à compléter avec vos classes)
        self.numeration_window = None
        self.laser_window = None
        self.ns_interface = None
        self.energy_window = None
        self.data_science_window = None
        
    def create_tool_button(self, text, icon):
        """Crée un bouton d'outil avec une icône et un texte"""
        button = QPushButton(f"  {icon}  {text}")
        button.setObjectName("main-button")
        button.setFont(QFont("Segoe UI", 12))
        button.setCursor(Qt.PointingHandCursor)
        
        # Effet d'ombre
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 3)
        button.setGraphicsEffect(shadow)
        
        return button
        
    def connect_buttons(self):
        """Connecte les boutons à leurs slots"""
        self.data_science_button.clicked.connect(self.open_data_science_window)
        self.energy_button.clicked.connect(self.open_energy_window)
        self.laser_button.clicked.connect(self.open_laser_window)
        self.navier_stokes_button.clicked.connect(self.open_navier_stokes_window)
        self.numerisation_button.clicked.connect(self.open_numeration_window)
        
    def open_numeration_window(self):
        if not self.numeration_window:
            # À remplacer par votre classe réelle
            self.numeration_window = QWidget()  
            self.numeration_window.setWindowTitle("Numérisation")
            self.numeration_window.resize(800, 600)
        self.numeration_window.show()
        
    def open_laser_window(self):
        if not self.laser_window:
            # À remplacer par votre classe réelle
            self.laser_window = QWidget()  
            self.laser_window.setWindowTitle("Simulation Laser")
            self.laser_window.resize(800, 600)
        self.laser_window.show()
        
    def open_energy_window(self):
        if not self.energy_window:
            # À remplacer par votre classe réelle
            self.energy_window = QWidget()  
            self.energy_window.setWindowTitle("Gestion Énergétique")
            self.energy_window.resize(800, 600)
        self.energy_window.show()
        
    def open_data_science_window(self):
        if not self.data_science_window:
            # À remplacer par votre classe réelle
            self.data_science_window = QWidget()  
            self.data_science_window.setWindowTitle("Data Science")
            self.data_science_window.resize(800, 600)
        self.data_science_window.show()
        
    def open_navier_stokes_window(self):
        if not self.ns_interface:
            
            self.ns_interface = QWidget()  
            self.ns_interface.setWindowTitle("Équations Navier-Stokes")
            self.ns_interface.resize(800, 600)
        self.ns_interface.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Configuration de la police par défaut
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
