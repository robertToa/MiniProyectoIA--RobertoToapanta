from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt

CARPETA = Path(__file__).resolve().parent / "rostros_cine"
CARPETA.mkdir(exist_ok=True)

# Detector facial incluido en opencv-python 4.x
DETECTOR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extraer_rostro(ruta):
    """Lee una foto y devuelve un rostro en gris, ajustado a 100x100."""
    imagen = cv2.imread(str(ruta))
    if imagen is None:
        raise ValueError("No se pudo abrir la imagen.")
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    caras = DETECTOR.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5)
    if len(caras) != 1:
        raise ValueError(f"Se necesita una foto con un solo rostro visible. Detectados: {len(caras)}.")
    x, y, w, h = caras[0]
    rostro = cv2.resize(gris[y:y+h, x:x+w], (100, 100))
    return cv2.equalizeHist(rostro)


def comparar_rostros(referencia, prueba):
    """Comparación sencilla para demostración, NO identificación fiable."""
    similitud = cv2.matchTemplate(referencia, prueba, cv2.TM_CCOEFF_NORMED)[0, 0]
    return float(similitud)


class Cine(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cine Uisrael 2026IA - Roberto Toapanta")
        self.setMinimumWidth(460)
        self.puesto = None
        self.botones = {}

        principal = QVBoxLayout(self)
        principal.setSpacing(14)

        titulo = QLabel("Cine Uisrael 2026IA - Roberto Toapanta")
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet("font-size: 17px; font-weight: bold;")
        principal.addWidget(titulo)

        pantalla = QLabel("PANTALLA")
        pantalla.setAlignment(Qt.AlignCenter)
        pantalla.setStyleSheet("background: #dce7f1; padding: 10px; border-radius: 5px;")
        principal.addWidget(pantalla)

        puestos = QHBoxLayout()
        for numero in range(1, 4):
            boton = QPushButton(f"Puesto {numero}")
            boton.setMinimumHeight(65)
            boton.clicked.connect(lambda _, n=numero: self.seleccionar(n))
            puestos.addWidget(boton)
            self.botones[numero] = boton
        principal.addLayout(puestos)

        self.estado = QLabel("Selecciona un puesto para comenzar.")
        self.estado.setAlignment(Qt.AlignCenter)
        principal.addWidget(self.estado)

        acciones = QHBoxLayout()
        cargar = QPushButton("Cargar imagen")
        validar = QPushButton("Validar biometría")
        cargar.clicked.connect(self.registrar)
        validar.clicked.connect(self.validar)
        cargar.setMinimumHeight(42)
        validar.setMinimumHeight(42)
        acciones.addWidget(cargar)
        acciones.addWidget(validar)
        principal.addLayout(acciones)

        nota = QLabel("Roberto Toapanta 2026 B")
        nota.setWordWrap(True)
        nota.setAlignment(Qt.AlignCenter)
        nota.setStyleSheet("color: #666; font-size: 11px;")
        principal.addWidget(nota)
        self.actualizar_botones()

    def actualizar_botones(self):
        for numero, boton in self.botones.items():
            registrado = (CARPETA / f"puesto_{numero}.npy").exists()
            color = "#a5d6a7" if registrado else "#e2e2e2"
            borde = "3px solid #2367b2" if numero == self.puesto else "1px solid #aaa"
            boton.setStyleSheet(f"background: {color}; border: {borde}; border-radius: 8px;")

    def seleccionar(self, numero):
        self.puesto = numero
        registrado = (CARPETA / f"puesto_{numero}.npy").exists()
        self.estado.setText(f"Puesto {numero} seleccionado — {'registrado' if registrado else 'disponible'}")
        self.actualizar_botones()

    def elegir_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar fotografía", "", "Imágenes (*.jpg *.jpeg *.png *.bmp)"
        )
        return ruta

    def registrar(self):
        if self.puesto is None:
            QMessageBox.information(self, "Puesto", "Selecciona primero un puesto.")
            return
        ruta = self.elegir_imagen()
        if not ruta:
            return
        try:
            rostro = extraer_rostro(ruta)
            np.save(CARPETA / f"puesto_{self.puesto}.npy", rostro)
            self.estado.setText(f"Puesto {self.puesto}: rostro registrado correctamente.")
            self.actualizar_botones()
            QMessageBox.information(self, "Registro", "Imagen facial registrada.")
        except ValueError as error:
            QMessageBox.warning(self, "Imagen no válida", str(error))

    def validar(self):
        if self.puesto is None:
            QMessageBox.information(self, "Puesto", "Selecciona primero un puesto.")
            return
        archivo = CARPETA / f"puesto_{self.puesto}.npy"
        if not archivo.exists():
            QMessageBox.warning(self, "Sin registro", "Primero registra una foto en este puesto.")
            return
        ruta = self.elegir_imagen()
        if not ruta:
            return
        try:
            referencia = np.load(archivo)
            prueba = extraer_rostro(ruta)
            similitud = comparar_rostros(referencia, prueba)
            # Umbral didáctico: NO calibrado para decisiones de acceso reales.
            coincide = similitud >= 0.75
            texto = "Coincidencia de imágenes" if coincide else "Imágenes diferentes"
            self.estado.setText(f"Puesto {self.puesto}: {texto} ({similitud:.2f})")
            QMessageBox.information(self, "Resultado", f"{texto}\nSimilitud: {similitud:.2f}\nResultado orientativo.")
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "Error", str(error))


if __name__ == "__main__":
    if DETECTOR.empty():
        raise SystemExit("No se pudo cargar el detector facial de OpenCV.")
    app = QApplication([])
    ventana = Cine()
    ventana.show()
    app.exec()
