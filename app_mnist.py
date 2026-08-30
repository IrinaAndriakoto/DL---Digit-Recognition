# -*- coding: utf-8 -*-
"""
=========================================================
 APPLICATION : Reconnaissance de chiffres dessinés à la souris
=========================================================
Ce programme est la partie 9 du TP « Digit Recognition ».
Il REUTILISE le modèle déployé par le notebook `tp_cnn_mnist.ipynb` :
    modele/cnn_mnist.pth

Fonctionnement :
    1. On redéfinit l'ARCHITECTURE du CNN (obligatoire : les poids seuls
       sont sauvegardés, pas l'objet complet).
    2. On charge les poids depuis le fichier `.pth` (équivalent joblib.load).
    3. Une fenêtre s'ouvre : on DESSINE un chiffre (blanc sur fond noir,
       comme les images MNIST), puis « Prédire » affiche la prédiction
       et la distribution de probabilités.

Lancement :
    python app_mnist.py
"""

import os

import torch
import torch.nn as nn

import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

# ======================================================================
# 1) ARCHITECTURE DU MODÈLE (identique à celle du notebook !)
# ======================================================================
class CNN(nn.Module):
    """Le MEILLEUR des CNN du notebook : ces couches DOIVENT correspondre
    exactement à celles utilisées à l'entraînement, sinon le chargement
    des poids échouera (erreur de dimensions)."""

    def __init__(self):
        super().__init__()
        # Blocs convolutionnels : extraction des caractéristiques
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        # Partie classification
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)          # logits (la CrossEntropy est déjà appliquée hors réseau)
        return x


# ======================================================================
# 2) CHARGEMENT DU MODÈLE DÉPLOYÉ  (équivalent de joblib.load)
# ======================================================================
def charger_modele(chemin="modele/cnn_mnist.pth"):
    """Charge l'architecture + les poids et renvoie le modèle prêt à prédire."""
    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}\n"
            "Exécutez d'abord le notebook tp_cnn_mnist.ipynb (Partie 8) "
            "pour entraîner et sauvegarder le modèle."
        )

    # GPU si disponible (sinon CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Reconstruire la même architecture
    model = CNN().to(device)
    # 2) Lire le dictionnaire sauvegardé et injecter les poids
    sauvegarde = torch.load(chemin, weights_only=True)
    model.load_state_dict(sauvegarde["state_dict"])
    # 3) Mode inférence : on désactive le dropout
    model.eval()

    acc_ref = sauvegarde.get("accuracy")
    print(f"Modèle chargé depuis : {chemin}  |  appareil : {device}")
    if acc_ref is not None:
        print(f"Accuracy de référence (test) : {acc_ref:.4f}")
    return model, device


# ======================================================================
# 3) PRÉDICTION D'UNE IMAGE (même prétraitement que l'entraînement)
# ======================================================================
def predire_chiffre(model, device, image_pil):
    """Prédit le chiffre d'une image PIL 280x280 (blanc sur noir).

    La chaîne de prétraitement doit reproduire EXACTEMENT celle du notebook :
        ToTensor()  -> division par 255 (pixels 0..1)
        Normalize((0.1307,), (0.3081,))
    Le dessin 280x280 est réduit à 28x28 (taille des images MNIST).
    """
    # --- Réduction à 28x28 (280/28 = 10 : les proportions sont conservées) ---
    img_28 = image_pil.resize((28, 28), Image.LANCZOS)

    # --- Même normalisation que pendant l'entraînement ---
    pixels = np.array(img_28, dtype=np.float32) / 255.0      # 0..1
    tenseur = torch.from_numpy(pixels)                       # (28,28)
    tenseur = (tenseur - 0.1307) / 0.3081                    # centrage/réduction
    tenseur = tenseur.unsqueeze(0).unsqueeze(0).to(device)   # (1,1,28,28)

    # --- Inférence ---
    with torch.no_grad():
        logits = model(tenseur)
        probas = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    chiffre_pred = int(probas.argmax())
    confidence   = float(probas[chiffre_pred])
    return chiffre_pred, confidence, probas


# ======================================================================
# 4) INTERFACE GRAPHIQUE (tkinter)
# ======================================================================
TAILLE_CANEVAS = 280   # 28 pixels MNIST x 10 (dessin grossi 10x)
EPAISSEUR = 14          # largeur du trait en pixels

class Application:
    def __init__(self, root, model, device):
        self.model = model
        self.device = device
        self.root = root
        root.title("Digit Recognition — CNN (TP Deep Learning)")
        root.configure(bg="#2b2b2b")

        # --- Image mémoire 280x280 : c'est elle qu'on analyse ---
        # (le canevas n'est qu'un affichage ; l'image est la source de vérité)
        self.image = Image.new("L", (TAILLE_CANEVAS, TAILLE_CANEVAS), 0)  # fond noir
        self.canvas_image = ImageDraw.Draw(self.image)
        self.dernier_point = None   # pour tracer des segments continus

        # --- Zone de dessin ---
        cadre = tk.Frame(root, bg="#2b2b2b")
        cadre.pack(padx=15, pady=15)

        lbl = tk.Label(cadre, text="✏️  Dessinez un chiffre (0-9) :",
                       bg="#2b2b2b", fg="white", font=("Segoe UI", 11))
        lbl.grid(row=0, column=0, columnspan=3, pady=(0, 8))

        self.canvas = tk.Canvas(cadre, width=TAILLE_CANEVAS,
                                height=TAILLE_CANEVAS, bg="black",
                                highlightthickness=1, highlightbackground="#666")
        self.canvas.grid(row=1, column=0, rowspan=4)
        self.canvas.bind("<B1-Motion>", self.dessiner)   # tracer en draguant
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "dernier_point", None))

        # --- Aperçu 28x28 (ce que le réseau « voit » réellement) ---
        lbl2 = tk.Label(cadre, text="Vue 28×28 du réseau :",
                        bg="#2b2b2b", fg="white", font=("Segoe UI", 10))
        lbl2.grid(row=1, column=1, padx=(15, 0))
        self.petit_canvas = tk.Canvas(cadre, width=140, height=140, bg="black",
                                      highlightthickness=1, highlightbackground="#555")
        self.petit_canvas.grid(row=2, column=1, padx=(15, 0))

        # --- Graphique des probabilités (barres) ---
        lbl3 = tk.Label(cadre, text="Probabilités :",
                        bg="#2b2b2b", fg="white", font=("Segoe UI", 10))
        lbl3.grid(row=1, column=2, padx=(15, 0))
        self.graphe = tk.Canvas(cadre, width=220, height=140, bg="#3c3f41",
                                highlightthickness=1, highlightbackground="#555")
        self.graphe.grid(row=2, column=2, padx=(15, 0))

        # --- Boutons ---
        btn_pred = tk.Button(cadre, text="Prédire", command=self.predire,
                             bg="#3a7d44", fg="white", font=("Segoe UI", 11, "bold"),
                             width=10)
        btn_pred.grid(row=3, column=1, columnspan=2, pady=(12, 4), padx=(15, 0))
        btn_eff  = tk.Button(cadre, text="Effacer", command=self.effacer,
                             bg="#7d3a3a", fg="white", font=("Segoe UI", 11),
                             width=10)
        btn_eff.grid(row=4, column=1, columnspan=2, padx=(15, 0))

        # --- Résultat ---
        self.resultat = tk.Label(root, text="", bg="#2b2b2b", fg="white",
                                 font=("Segoe UI", 14, "bold"))
        self.resultat.pack(pady=(0, 15))

        self.afficher_probas(np.zeros(10))   # le graphe vide au démarrage

    # --------------------------------------------------------------
    def dessiner(self, event):
        """Trace un trait entre le dernier point et la position actuelle,
        à la fois sur le canevas (visuel) et sur l'image mémoire 280x280."""
        x, y = event.x, event.y
        rayon = EPAISSEUR // 2
        self.canvas.create_oval(x - rayon, y - rayon, x + rayon, y + rayon,
                                fill="white", outline="white")
        if self.dernier_point is not None:
            px, py = self.dernier_point
            # largeur EPAISSEUR : trait « épais » de style feutre
            self.canvas_image.line([px, py, x, y], fill=255, width=EPAISSEUR)
        else:
            self.canvas_image.ellipse([x - rayon, y - rayon, x + rayon, y + rayon],
                                      fill=255)
        self.dernier_point = (x, y)

    # --------------------------------------------------------------
    def predire(self):
        """Lance la prédiction et met à jour l'affichage."""
        chiffre, conf, probas = predire_chiffre(self.model, self.device, self.image)
        self.afficher_resultat(chiffre, conf, probas)
        self.afficher_vue28()

    # --------------------------------------------------------------
    def afficher_resultat(self, chiffre, conf, probas):
        self.resultat.config(
            text=f"Chiffre prédit :  {chiffre}   (confiance {conf:.1%})",
            fg="#7CFC00")

        # Top 3 pour l'analyse (transparence du modèle)
        print(f"Prédiction : {chiffre} (confiance {conf:.2%})")
        print("  " + ", ".join(
            f"{i}:{p:.1%}" for i, p in enumerate(probas)))

        self.afficher_probas(probas)

    # --------------------------------------------------------------
    def afficher_probas(self, probas):
        """Dessine un histogramme horizontal des 10 probabilités."""
        self.graphe.delete("all")
        h = 140
        for i, p in enumerate(probas):
            y = i * (h / 10) + 2
            largeur = max(2, int(p * 190))
            couleur = "#4fc3f7" if p < max(probas) else "#7CFC00"
            self.graphe.create_rectangle(2, y, 2 + largeur, y + h / 10 - 3,
                                         fill=couleur, outline="")
            self.graphe.create_text(210, y + h / 20, text=str(i), fill="white")

    # --------------------------------------------------------------
    def afficher_vue28(self):
        """Montre l'image réduite à 28x28 (vue réelle du réseau)."""
        img_28 = self.image.resize((28, 28), Image.LANCZOS)
        img_28 = img_28.resize((140, 140), Image.NEAREST)   # agrandie pour l'affichage
        photo = Image.fromarray(np.array(img_28, dtype=np.uint8))
        from PIL import ImageTk
        self.photo = ImageTk.PhotoImage(photo)   # on garde une référence !
        self.petit_canvas.delete("all")
        self.petit_canvas.create_image(70, 70, image=self.photo)

    # --------------------------------------------------------------
    def effacer(self):
        """Remet le canevas et l'image mémoire à zéro."""
        self.image = Image.new("L", (TAILLE_CANEVAS, TAILLE_CANEVAS), 0)
        self.canvas_image = ImageDraw.Draw(self.image)
        self.canvas.delete("all")
        self.petit_canvas.delete("all")
        self.resultat.config(text="", fg="white")
        self.afficher_probas(np.zeros(10))

    # --------------------------------------------------------------
    def demarrer(self):
        self.root.mainloop()


# ======================================================================
# POINT D'ENTRÉE
# ======================================================================
if __name__ == "__main__":
    modele, device = charger_modele()          # charge le modèle déployé
    racine = tk.Tk()                            # fenêtre principale
    Application(racine, modele, device).demarrer()