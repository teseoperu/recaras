import os
import json
import faiss
import numpy as np
import cv2
import insightface
import tkinter as tk
from tkinter import filedialog
import shutil

def seleccionar_carpeta(mensaje="Selecciona una carpeta"):
    root = tk.Tk()
    root.withdraw()
    carpeta = filedialog.askdirectory(title=mensaje)
    return carpeta

def seleccionar_archivo(mensaje="Selecciona una imagen"):
    root = tk.Tk()
    root.withdraw()
    archivo = filedialog.askopenfilename(
        title=mensaje,
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
    )
    return archivo

def extraer_embeddings(model, image_path):
    img = cv2.imread(image_path)
    if img is None:
        return []
    faces = model.get(img)
    if not faces:
        return []
    return [face.normed_embedding for face in faces]

def main():
    # Paso 1: Seleccionar carpeta de índice
    carpeta_indice = seleccionar_carpeta("Selecciona la carpeta *_index con el índice FAISS")
    if not carpeta_indice:
        print("❌ No seleccionaste carpeta de índice.")
        return

    faiss_path = os.path.join(carpeta_indice, "index.faiss")
    json_path = os.path.join(carpeta_indice, "index.json")

    if not os.path.exists(faiss_path) or not os.path.exists(json_path):
        print("❌ Archivos de índice no encontrados en la carpeta seleccionada.")
        return

    # Paso 2: Seleccionar foto de consulta
    query_path = seleccionar_archivo("Selecciona la foto de consulta")
    if not query_path:
        print("❌ No seleccionaste una foto.")
        return

    # Paso 3: Umbral de similitud
    try:
        umbral = float(input("👉 Ingresa umbral de similitud (ej. 0.8): "))
    except ValueError:
        print("❌ Umbral inválido.")
        return

    # Paso 4: Cargar modelo
    print("⚙️ Cargando modelo InsightFace...")
    model = insightface.app.FaceAnalysis(name="antelopev2")
    model.prepare(ctx_id=0)

    # Paso 5: Cargar índice FAISS y metadatos
    print("📂 Cargando índice...")
    index = faiss.read_index(faiss_path)
    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Paso 6: Extraer embedding de la foto de consulta
    query_embs = extraer_embeddings(model, query_path)
    if not query_embs:
        print("❌ No se detectaron rostros en la foto de consulta.")
        return

    query_embs = np.array(query_embs).astype("float32")

    # Paso 7: Buscar coincidencias
    print(f"🔍 Buscando coincidencias con umbral {umbral}...")
    D, I = index.search(query_embs, k=200)  # top 10 vecinos por cara

    # Carpeta de resultados
    nombre_base = os.path.splitext(os.path.basename(query_path))[0]
    carpeta_resultados = os.path.join("resultados", nombre_base)
    os.makedirs(carpeta_resultados, exist_ok=True)

    # Paso 8: Evaluar resultados y copiar imágenes
    encontrados = set()
    for i, vecinos in enumerate(I):
        for j, idx in enumerate(vecinos):
            if idx < 0:  # sin vecino
                continue
            dist = D[i][j]
            score = 1 / (1 + dist)  # convertir distancia L2 a score aproximado
            if score >= umbral:
                meta = metadata[idx]
                origen = meta["imagen"]
                if origen not in encontrados:
                    encontrados.add(origen)
                    destino = os.path.join(carpeta_resultados, os.path.basename(origen))
                    shutil.copy(origen, destino)
                    print(f"✅ Coincidencia: {origen} (score {score:.4f})")

    print(f"📂 Resultados copiados en: {carpeta_resultados}")
    print(f"🔢 Total de coincidencias: {len(encontrados)}")

if __name__ == "__main__":
    main()

