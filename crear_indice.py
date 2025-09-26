import os
import glob
import faiss
import json
import numpy as np
import cv2
import insightface
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
import signal
import sys

stop_requested = False

def signal_handler(sig, frame):
    global stop_requested
    print("\n🛑 Interrupción detectada. Guardando progreso...")
    stop_requested = True

signal.signal(signal.SIGINT, signal_handler)

def seleccionar_carpeta():
    """Muestra un cuadro de diálogo para elegir la carpeta de imágenes."""
    root = tk.Tk()
    root.withdraw()
    carpeta = filedialog.askdirectory(title="Selecciona la carpeta con las fotos")
    return carpeta

def extraer_embeddings(model, image_path):
    """Extrae los embeddings de todas las caras encontradas en la imagen."""
    img = cv2.imread(image_path)
    if img is None:
        return []
    faces = model.get(img)
    if not faces:
        return []
    return [face.normed_embedding for face in faces]

def main():
    # Paso 1: Selección de carpeta
    carpeta = seleccionar_carpeta()
    if not carpeta:
        print("No seleccionaste ninguna carpeta. Saliendo...")
        return

    print(f"📂 Procesando imágenes en: {carpeta}")

    # Paso 2: Crear carpeta de salida
    nombre_carpeta = os.path.basename(carpeta.rstrip(os.sep))
    carpeta_salida = os.path.join(os.path.dirname(carpeta), nombre_carpeta + "_index")
    os.makedirs(carpeta_salida, exist_ok=True)

    faiss_path = os.path.join(carpeta_salida, "index.faiss")
    json_path = os.path.join(carpeta_salida, "index.json")
    progress_path = os.path.join(carpeta_salida, "progress.json")

    # Paso 3: Cargar modelo
    print("⚙️ Cargando modelo InsightFace...")
    model = insightface.app.FaceAnalysis(name="antelopev2")
    model.prepare(ctx_id=0)

    # Paso 4: Buscar imágenes
    extensiones = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in extensiones:
        image_paths.extend(glob.glob(os.path.join(carpeta, "**", ext), recursive=True))
    image_paths = sorted(image_paths)

    if not image_paths:
        print("❌ No se encontraron imágenes en la carpeta.")
        return

    # Paso 5: Cargar progreso previo si existe
    embeddings = []
    metadata = []
    procesadas = set()

    if os.path.exists(progress_path) and os.path.exists(faiss_path) and os.path.exists(json_path):
        print("📂 Reanudando desde progreso previo...")
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
            procesadas = set(progress.get("procesadas", []))

        # Cargar FAISS y metadatos
        index = faiss.read_index(faiss_path)
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        print("🆕 Creando un nuevo índice...")
        index = None

    # Paso 6: Procesar imágenes faltantes
    faltantes = [p for p in image_paths if p not in procesadas]
    print(f"🔍 {len(faltantes)} imágenes por procesar (de {len(image_paths)} totales).")

    try:
        for img_path in tqdm(faltantes):
            if stop_requested:
                break

            embs = extraer_embeddings(model, img_path)
            if not embs:
                continue

            embs = np.array(embs).astype("float32")

            # Crear índice si aún no existe
            if index is None:
                dim = embs.shape[1]
                index = faiss.IndexFlatL2(dim)

            index.add(embs)

            for i in range(len(embs)):
                metadata.append({"imagen": img_path, "cara_id": i})

            # Marcar como procesada
            procesadas.add(img_path)

            # Guardar progreso parcial
            faiss.write_index(index, faiss_path)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump({"procesadas": list(procesadas)}, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"❌ Error durante el procesamiento: {e}")

    # Paso 7: Mensaje final
    if stop_requested:
        print("⚠️ Proceso interrumpido. Puedes reanudar más tarde.")
    else:
        print("✅ Proceso completado. Índice listo.")

if __name__ == "__main__":
    main()
