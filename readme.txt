==========================================
README - Programas de Reconocimiento Facial
==========================================

Descripción General
-------------------
Este proyecto contiene dos programas escritos en Python que permiten analizar una carpeta con fotos,
detectar rostros y luego buscar todas las imágenes donde aparece una persona específica.

Programas Incluidos
-------------------

1. crear_indice.py
   ----------------
   Este programa sirve para analizar una carpeta de fotos y generar un índice de rostros que se usará
   en búsquedas posteriores.

   Funciones principales:
   - Permite seleccionar la carpeta de fotos a procesar.
   - Recorre todas las imágenes en formatos .jpg, .jpeg, .png, .bmp.
   - Usa el modelo InsightFace para detectar rostros en cada foto.
   - Convierte cada rostro en un vector numérico (embedding).
   - Guarda la información en un índice mediante la librería FAISS.
   - Almacena un archivo index.json con los metadatos (rostros e imágenes).
   - Genera un archivo progress.json para continuar si el proceso se interrumpe.

   Resultado:
   - Se crea una carpeta *_index con:
     * index.faiss -> contiene el índice de rostros.
     * index.json  -> lista de imágenes y rostros detectados.
     * progress.json -> para reanudar el procesamiento.

2. buscar_persona.py
   ------------------
   Este programa permite buscar todas las fotos donde aparece la misma persona a partir de una foto
   de consulta.

   Funciones principales:
   - Permite seleccionar la carpeta *_index creada con el primer programa.
   - Solicita una foto de consulta (la persona que se desea buscar).
   - Pide un umbral de similitud (ejemplo: 0.8).
   - Carga el índice y los metadatos del primer programa.
   - Detecta el rostro en la foto de consulta y lo compara contra todos los rostros del índice.
   - Copia en la carpeta resultados/ todas las fotos coincidentes.

   Resultado:
   - Se genera una carpeta resultados/ con el nombre de la foto de consulta.
   - Dentro se guardan las fotos donde aparece la persona buscada.
   - En la terminal se muestra la lista de coincidencias encontradas con su nivel de similitud.

Relación entre los Programas
----------------------------
1. Primero se ejecuta crear_indice.py para analizar las fotos y construir el índice.
2. Luego se ejecuta buscar_persona.py para encontrar en qué fotos aparece la persona de la foto de consulta.

De esta manera, el sistema permite organizar y buscar personas en grandes colecciones de fotos de forma rápida y eficiente.
