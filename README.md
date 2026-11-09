# Aplicaciones de Inteligencia Artificial · Segunda evaluación

## Detección de correos fraudulentos con una red LSTM

Un modelo de Deep Learning lee el texto de un correo y estima la probabilidad de que sea phishing. Con esa probabilidad, el área de TI puede enviar los correos sospechosos a cuarentena y revisarlos antes de que lleguen a la bandeja de entrada.

- **Video:** (pendiente)
- **Integrantes:** [Nombre estudiante 1] (problema y modelo) · [Nombre estudiante 2] (implementación en vivo)

## Contenido

| Archivo | Qué es |
|---|---|
| [`deteccion_phishing_lstm.ipynb`](segunda-evaluacion/deteccion_phishing_lstm.ipynb) | Código fuente: notebook completo, ejecutable en Google Colab |
| [`presentacion_estudiante_1.pdf`](segunda-evaluacion/presentacion_estudiante_1.pdf) | Presentación del estudiante 1: problema y modelo (PDF) |
| [`presentacion_estudiante_1.pptx`](segunda-evaluacion/presentacion_estudiante_1.pptx) | La misma presentación, editable en PowerPoint o Google Slides |
| [`infografia_estudiante_1.pdf`](segunda-evaluacion/infografias/infografia_estudiante_1.pdf) | Infografía individual del estudiante 1 para la defensa: problema y modelo |
| [`infografia_estudiante_2.pdf`](segunda-evaluacion/infografias/infografia_estudiante_2.pdf) | Infografía individual del estudiante 2 para la defensa: implementación y resultados |
| [`infografias_estudiantes_1_y_2.pdf`](segunda-evaluacion/infografias/infografias_estudiantes_1_y_2.pdf) | Las dos infografías en un solo PDF (página 1: estudiante 1, página 2: estudiante 2) |
| [`guia_de_estudio.pdf`](segunda-evaluacion/guia_de_estudio.pdf) | Guía de estudio para la defensa: conceptos, explicaciones y 36 preguntas con respuesta |
| [`dialogo_estudiante_1.md`](segunda-evaluacion/dialogo_estudiante_1.md) | Diálogo del estudiante 1, lámina por lámina |
| [`guia_estudiante_1_contexto_y_modelo.md`](segunda-evaluacion/guia_estudiante_1_contexto_y_modelo.md) | Ideas clave del estudiante 1 y preguntas probables de la defensa |
| [`phishing_en_vivo.ipynb`](segunda-evaluacion/phishing_en_vivo.ipynb) | Notebook con los títulos de cada paso, para escribir el código en vivo en Colab |
| [`dialogo_estudiante_2.md`](segunda-evaluacion/dialogo_estudiante_2.md) | Diálogo del estudiante 2: qué decir mientras escribe cada bloque de código |
| [`guia_phishing_en_vivo.md`](segunda-evaluacion/guia_phishing_en_vivo.md) | Guía de programación en vivo del estudiante 2 |
| [`decisiones_del_proyecto.md`](segunda-evaluacion/decisiones_del_proyecto.md) | Por qué tomamos cada decisión del proyecto |

## Cómo ejecutarlo

1. Abrir el notebook en Google Colab (*Archivo → Subir notebook*).
2. Opcional: activar la GPU (*Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*). No es necesaria.
3. Ejecutar todas las celdas. El dataset se descarga con `kagglehub` y no necesita cuenta de Kaggle. Sin GPU, el entrenamiento toma alrededor de un minuto.

También funciona en un computador local, instalando `tensorflow scikit-learn pandas matplotlib kagglehub`.

## Datos y modelo

- **Datos:** Phishing Email Detection (Kaggle, `subhajournal/phishingemails`): 18.650 correos etiquetados como seguros o fraudulentos, 17.103 únicos después de la limpieza. Aprendizaje supervisado.
- **Modelo:** Embedding (20.000 × 64) → LSTM (64) → Dropout (0,5) → Dense (1, sigmoide), entrenado con entropía cruzada binaria y Adam.
- **Comparación:** regresión logística sobre TF-IDF, para medir si la LSTM aporta frente a un modelo simple.

## Resultados en los datos de prueba (3.421 correos)

| Métrica | LSTM | Regresión logística |
|---|---|---|
| Exactitud | 0,967 | 0,967 |
| Recall de fraude | 0,945 | 0,940 |
| AUC | 0,991 | 0,995 |

La LSTM detectó 1.218 de 1.289 correos fraudulentos y marcó por error 41 de 2.132 correos seguros. En este dataset no supera al modelo simple. La solución responde al problema de forma parcial: sirve como filtro de apoyo contra el fraude evidente, pero dejó pasar un fraude bien escrito y no reemplaza otros controles.

Los valores provienen de la ejecución del notebook y pueden variar levemente entre ejecuciones y entre CPU y GPU.
