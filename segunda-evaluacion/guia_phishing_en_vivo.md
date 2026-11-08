# Guía de programación en vivo: correos fraudulentos con LSTM

Rol: **integrante 3** (programación, ejecución y demostración). Máximo en el video: **15 minutos**.

El enunciado no permite solo ejecutar un notebook terminado. Esta guía sirve para escribir el código paso a paso, explicándolo. `deteccion_phishing_lstm.ipynb` tiene el mismo código y queda de respaldo.

## Antes de grabar

- [ ] Subir `phishing_en_vivo.ipynb` a Colab (*Archivo → Subir notebook*) y activar la GPU (*Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*). Detalle en `dialogo_estudiante_2.md`, sección *Antes de grabar*.
- [ ] Ensayar completo y cronometrar en una copia del notebook. **Todavía no se ha probado en Colab**, solo en un computador local.
- [ ] Tener esta guía a la vista, pero explicar con palabras propias: la rúbrica castiga la lectura.
- [ ] Aumentar el zoom del navegador para que el código se lea en el video.
- [ ] Al terminar, ejecutar el paso 17 y guardar `modelo_phishing.keras` para la defensa.

## Qué se escribe en vivo y qué se puede pegar

| Paso | Modo | Min. |
|---|---|---|
| Paso 1. Preparar el entorno | Escribir | 0,5 |
| Paso 2. Importar las bibliotecas | Escribir | 0,5 |
| Paso 3. Cargar los datos | Escribir | 0,5 |
| Paso 4. Explorar la estructura de los datos | Escribir | 1,5 |
| Paso 5. Limpiar el texto | Escribir | 1,5 |
| Paso 6. Separar los datos | Escribir | 1 |
| Paso 7. Convertir las palabras en números | Escribir | 1 |
| Paso 8. Crear una línea de comparación | Escribir | 1 |
| Paso 9. Construir la arquitectura LSTM | Escribir | 1,5 |
| Paso 10. Configurar el entrenamiento | Escribir | 0,5 |
| Paso 11. Entrenar el modelo | Escribir | 2 |
| Paso 12. Curvas de aprendizaje | Pegar | 0,5 |
| Paso 13. Evaluar y comparar | Escribir | 1 |
| Paso 14. Probar con correos nuevos | Escribir la función, pegar los correos | 1,5 |
| Paso 15. Interpretar el resultado | Escribir | 0,5 |
| Paso 16 (opcional). Efecto del umbral | Escribir | Fuera del tiempo |
| Paso 17. Guardar el modelo (para la defensa) | Escribir | Fuera del tiempo |

Los pasos con tiempo suman unos **15,0 minutos** estimados. Es una estimación: escribir en vivo suele demorar más, así que hay que cronometrar el ensayo. Si queda largo, se puede pasar más rápido por los pasos 2 y 12.

## Correspondencia con la rúbrica (integrante 3)

| Dimensión | Dónde se cumple |
|---|---|
| 1. Entorno, bibliotecas y datos | Pasos 1 a 4 |
| 2. Preprocesamiento y construcción del modelo | Pasos 5 a 7 y 9 |
| 3. Programación y explicación del código | Todos: escribir en vivo y explicar por qué |
| 4. Entrenamiento, métricas y funcionamiento | Pasos 10 a 14 |
| 5. Interpretación y relación con el problema | Pasos 8, 13, 15 y cierre |

---

## Paso 1. Preparar el entorno · Escribir · ~0,5 min

```python
# Paso 1: preparar el entorno
%pip install -q kagglehub
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPU disponible:", tf.config.list_physical_devices('GPU'))
```

**Qué explicar**

- Me identifico: nombre e integrante 3, a cargo de la implementación.
- Trabajo en Google Colab con GPU: la LSTM repite muchos cálculos por cada palabra y la GPU los acelera. Colab ya trae TensorFlow; con `%pip` me aseguro de tener `kagglehub`.
- `kagglehub` me permite descargar el dataset directamente desde Kaggle.

**Por qué así** (para responder preguntas)

- Sin GPU el código funciona igual. En mi ensayo en CPU, el entrenamiento tomó poco más de 1 minuto.

**Resultado esperado**

La versión de TensorFlow y la lista de GPU.

## Paso 2. Importar las bibliotecas · Escribir · ~0,5 min

```python
# Paso 2: importar bibliotecas
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, ConfusionMatrixDisplay)

keras.utils.set_random_seed(42)   # resultados repetibles
```

**Qué explicar**

- Digo para qué sirve cada biblioteca, sin leer la lista completa.
- La semilla fija los valores aleatorios para poder repetir el experimento.

**Por qué así** (para responder preguntas)

- La misma estructura de `Sequential` aparece en los ejemplos del curso.

**Resultado esperado**

Sin salida.

## Paso 3. Cargar los datos · Escribir · ~0,5 min

```python
# Paso 3: cargar los datos
ruta = kagglehub.dataset_download('subhajournal/phishingemails')
datos = pd.read_csv(f'{ruta}/Phishing_Email.csv')
datos = datos.rename(columns={'Email Text': 'texto', 'Email Type': 'tipo'})[['texto', 'tipo']]
datos.head()
```

**Qué explicar**

- Es aprendizaje **supervisado**: cada correo trae la respuesta correcta.
- Me quedo con dos columnas: el texto (entrada) y el tipo (lo que quiero predecir).

**Por qué así** (para responder preguntas)

- El archivo pesa 52 MB y en mi ensayo se descargó sin cuenta de Kaggle.

**Resultado esperado**

Una tabla con las primeras 5 filas.

## Paso 4. Explorar la estructura de los datos · Escribir · ~1,5 min

```python
# Paso 4: explorar la estructura de los datos
print("Filas:", len(datos), "| Texto vacío:", datos['texto'].isna().sum())
print("Duplicados exactos:", datos['texto'].duplicated().sum())
print(datos['tipo'].value_counts())

palabras = datos['texto'].dropna().str.split().str.len()
print("Mediana de palabras por correo:", int(palabras.median()),
      "| 90% de los correos tiene menos de:", int(palabras.quantile(0.9)))

con_enron = datos['texto'].fillna('').str.lower().str.contains(r'\benron\b')
print("\n% de correos que dicen 'enron', por tipo:")
print((con_enron.groupby(datos['tipo']).mean() * 100).round(1))
```

**Qué explicar**

- Hay duplicados y textos vacíos: los voy a limpiar, porque un correo repetido podría quedar en entrenamiento y en prueba, e inflar el resultado.
- Hay más correos seguros que fraudulentos, pero el desbalance es moderado, así que no basta con mirar exactitud y también voy a medir recall y precisión.
- **Hallazgo clave:** 'enron' es el nombre de una empresa, y aparece en cerca del 20% de los correos seguros y prácticamente en ningún correo fraudulento. Si lo dejo, el modelo aprende un atajo: 'dice enron, entonces es seguro'. Eso no sirve en ninguna otra empresa.

**Por qué así** (para responder preguntas)

- Por eso se revisa antes de entrenar: un atajo así puede dar buen resultado aquí y no servir afuera.

**Resultado esperado**

```
Filas: 18650 | Texto vacío: 16
Duplicados exactos: 1112
Safe Email 11322 | Phishing Email 7328
Mediana de palabras por correo: 160
% de correos que dicen 'enron': Phishing 0.0 | Safe 20.3
```

## Paso 5. Limpiar el texto · Escribir · ~1,5 min

```python
# Paso 5: limpiar el texto
def limpiar(texto):
    texto = str(texto).lower()
    texto = re.sub(r'<[^>]+>', ' ', texto)                           # etiquetas HTML
    texto = re.sub(r'(https?://|www\.)\S+', ' enlaceweb ', texto)    # enlaces
    texto = re.sub(r'\b(enron|vince|kaminski|hou|ect)\b', ' ', texto)  # atajos del dataset
    texto = re.sub(r'[^a-záéíóúñü ]', ' ', texto)                    # solo letras
    return re.sub(r'\s+', ' ', texto).strip()

datos = datos.dropna(subset=['texto'])
datos['limpio'] = datos['texto'].map(limpiar)
datos = datos[datos['limpio'] != ''].drop_duplicates('limpio')
print("Correos únicos:", len(datos))
print(datos['tipo'].value_counts())
print("\nEjemplo:", datos['limpio'].iloc[0][:150])
```

**Qué explicar**

- Explico cada línea de `limpiar` y qué problema evita.
- Elimino duplicados después de limpiar, porque dos correos pueden volverse iguales solo después de la limpieza.
- Esta misma función se usará con los correos nuevos: el modelo debe recibir siempre el mismo formato.

**Por qué así** (para responder preguntas)

- `re.sub` busca un patrón y lo reemplaza. `\b` significa borde de palabra: así no se borra 'ect' dentro de 'project'.
- Hallazgo para la defensa: después de limpiar, `enlaceweb` aparece en el 30,9% de los correos seguros y en el 10,7% de los fraudes. Una causa probable: muchas URL del dataset vienen partidas con espacios ('http : / / ...') y la regla no las detecta. En este dataset, tener un enlace inclina hacia 'seguro'. Mejora: una regla que capture también esas URL.
- Una limitación: el dataset mezcla phishing con spam común, así que hablo de correos fraudulentos y spam.

**Resultado esperado**

```
Correos únicos: 17103
Safe Email 10661 | Phishing Email 6442
```

## Paso 6. Separar los datos · Escribir · ~1 min

```python
# Paso 6: separar entrenamiento, validación y prueba
X = datos['limpio'].values
y = (datos['tipo'] == 'Phishing Email').astype(int).values   # 1 = fraudulento, 0 = seguro

X_resto, X_prueba, y_resto, y_prueba = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
X_ent, X_val, y_ent, y_val = train_test_split(
    X_resto, y_resto, test_size=0.15, stratify=y_resto, random_state=42)
print("Entrenamiento:", len(X_ent), "| Validación:", len(X_val), "| Prueba:", len(X_prueba))
```

**Qué explicar**

- Entrenamiento: la red aprende. Validación: vigilo el sobreajuste. Prueba: evaluación final con correos que nunca vio.
- Convierto la etiqueta a 0 y 1 porque la red trabaja con números.

**Por qué así** (para responder preguntas)

- Separo después de quitar duplicados: si lo hiciera antes, el mismo correo podría quedar en ambos lados.

**Resultado esperado**

Entrenamiento: 11629 | Validación: 2053 | Prueba: 3421

## Paso 7. Convertir las palabras en números · Escribir · ~1 min

```python
# Paso 7: convertir las palabras en números
VOCABULARIO, LARGO = 20000, 200
vectorizador = layers.TextVectorization(
    max_tokens=VOCABULARIO, output_sequence_length=LARGO, standardize=None)
vectorizador.adapt(X_ent)   # aprende el vocabulario solo con entrenamiento

X_ent_num = vectorizador(X_ent)
X_val_num = vectorizador(X_val)
X_prueba_num = vectorizador(X_prueba)
print("Forma:", X_ent_num.shape)
print("Correo:", X_ent[0][:60])
print("Números:", X_ent_num[0][:10].numpy())
```

**Qué explicar**

- Una red no entiende texto: cada palabra se reemplaza por su número en el vocabulario.
- `adapt` usa solo entrenamiento. Si usara la prueba, le estaría filtrando información al modelo.
- `standardize=None` porque ya limpié el texto con mi función.

**Por qué así** (para responder preguntas)

- 200 palabras es un equilibrio: cubre más de la mediana de los correos y mantiene el entrenamiento rápido. Las palabras después de la 200 no se leen, y esa es una limitación.

**Resultado esperado**

Forma `(n, 200)` y un ejemplo de correo con sus números.

## Paso 8. Crear una línea de comparación · Escribir · ~1 min

```python
# Paso 8: línea de comparación (regresión logística)
tfidf = TfidfVectorizer(max_features=VOCABULARIO)
base = LogisticRegression(max_iter=1000).fit(tfidf.fit_transform(X_ent), y_ent)
prob_base = base.predict_proba(tfidf.transform(X_prueba))[:, 1]
print(classification_report(y_prueba, (prob_base >= 0.5).astype(int),
                            target_names=['Seguro', 'Fraudulento'], digits=3))
```

**Qué explicar**

- Un modelo complejo solo se justifica si aporta frente a uno simple, así que primero mido el simple.
- TF-IDF da más peso a las palabras distintivas de cada correo. La regresión combina esos pesos para estimar la probabilidad de fraude.

**Por qué así** (para responder preguntas)

- Es lo que permite responder con datos a la pregunta '¿por qué no un modelo más simple?'.

**Resultado esperado**

Exactitud cerca de 0,967 | recall de fraude cerca de 0,94

## Paso 9. Construir la arquitectura LSTM · Escribir · ~1,5 min

```python
# Paso 9: construir la arquitectura
modelo = keras.Sequential([
    keras.Input(shape=(LARGO,)),
    layers.Embedding(VOCABULARIO, 64, mask_zero=True),  # ignora el relleno de ceros
    layers.LSTM(64),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
modelo.summary()
```

**Qué explicar**

- Es la arquitectura que justificó el integrante 2: un correo es una **secuencia**, y una LSTM puede considerar el orden, por ejemplo 'click the link below to confirm your password'.
- `mask_zero=True` le indica a la red que los ceros son relleno, no palabras.
- La salida es una neurona con sigmoide porque el problema es binario.
- En el `summary`, casi todos los parámetros están en el Embedding: 20.000 palabras por 64 valores, 1,28 millones.

**Por qué así** (para responder preguntas)

- Si aparece un aviso sobre `mask`, indica que las capas siguientes no reciben la máscara. La LSTM la usa igual (lo comprobé en mi ensayo: el resultado es igual a leer solo las palabras reales, sin relleno).
- Usa los componentes del ejemplo de noticias falsas del curso (Embedding de 64, LSTM, Dropout de 0,5 y Dense con sigmoide), pero con una sola LSTM en vez de dos apiladas: es más simple y más rápida de entrenar en vivo. No probamos la versión con dos LSTM.

**Resultado esperado**

Tabla `summary` con 1.313.089 parámetros.

## Paso 10. Configurar el entrenamiento · Escribir · ~0,5 min

```python
# Paso 10: configurar el entrenamiento
modelo.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy',
             keras.metrics.Recall(name='recall'),
             keras.metrics.Precision(name='precision')]
)
```

**Qué explicar**

- La pérdida mide el error; Adam usa su gradiente, calculado con retropropagación, para ajustar cada peso.
- Mido recall porque un fraude no detectado puede terminar en robo de claves, y precisión porque una falsa alarma puede esconder un correo legítimo.

**Por qué así** (para responder preguntas)

- Binary cross-entropy es la pérdida que corresponde a una salida sigmoide.

**Resultado esperado**

Sin salida.

## Paso 11. Entrenar el modelo · Escribir · ~2 min

```python
# Paso 11: entrenar el modelo
parada = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=2, restore_best_weights=True)
historial = modelo.fit(
    X_ent_num, y_ent,
    validation_data=(X_val_num, y_val),
    epochs=6,
    batch_size=32,
    callbacks=[parada],
    verbose=2
)
```

**Qué explicar**

- Mientras corre explico el ciclo: la red lee 32 correos, calcula la pérdida, retropropaga el error y Adam actualiza los pesos.
- `loss` es de entrenamiento y `val_loss` de correos que no se usan para ajustar pesos. Si `loss` baja y `val_loss` sube, hay sobreajuste, y ahí actúa EarlyStopping.

**Por qué así** (para responder preguntas)

- EarlyStopping no aparece en el material del curso. Se puede explicar con los conceptos de época y sobreajuste, que sí pide el enunciado.
- La validación puede saltar bastante entre épocas: el entrenamiento no es perfectamente estable.

**Resultado esperado**

Una línea por época. En el ensayo se detuvo en la época 4 y recuperó la 2 (val_accuracy 0,969). Tardó poco más de 1 minuto en CPU; en GPU aún no está medido.

## Paso 12. Curvas de aprendizaje · Pegar · ~0,5 min

```python
# Paso 12: curvas de aprendizaje
h = historial.history
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5))
a1.plot(h['loss'], label='Entrenamiento'); a1.plot(h['val_loss'], label='Validación')
a1.set_title('Pérdida'); a1.set_xlabel('Época'); a1.legend()
a2.plot(h['accuracy'], label='Entrenamiento'); a2.plot(h['val_accuracy'], label='Validación')
a2.set_title('Exactitud'); a2.set_xlabel('Época'); a2.legend()
plt.show()
```

**Qué explicar**

- Comento los números que salieron, en especial si la validación se separa del entrenamiento.

**Por qué así** (para responder preguntas)

- Es código auxiliar: se puede pegar, pero se explica igual.

**Resultado esperado**

Dos gráficos.

## Paso 13. Evaluar y comparar · Escribir · ~1 min

```python
# Paso 13: evaluar y comparar
prob_lstm = modelo.predict(X_prueba_num, verbose=0).ravel()
pred_lstm = (prob_lstm >= 0.5).astype(int)
print(classification_report(y_prueba, pred_lstm,
                            target_names=['Seguro', 'Fraudulento'], digits=3))
print("AUC LSTM:", round(roc_auc_score(y_prueba, prob_lstm), 3),
      "| AUC regresión logística:", round(roc_auc_score(y_prueba, prob_base), 3))
print("Exactitud LSTM:", round((pred_lstm == y_prueba).mean(), 3),
      "| regresión logística:", round(((prob_base >= 0.5).astype(int) == y_prueba).mean(), 3))
ConfusionMatrixDisplay(confusion_matrix(y_prueba, pred_lstm),
                       display_labels=['Seguro', 'Fraudulento']).plot(cmap='Blues')
plt.show()
```

**Qué explicar**

- Leo la matriz: la diagonal son aciertos. Abajo a la izquierda están los fraudes no detectados, y arriba a la derecha las falsas alarmas.
- **Lo digo sin rodeos:** en este dataset la LSTM rinde prácticamente igual que la regresión logística. Con estos datos, las palabras por sí solas ya separan muy bien las clases.

**Por qué así** (para responder preguntas)

- Si preguntan '¿entonces para qué la LSTM?': en este dataset no mostró una mejora medible. Su ventaja teórica es considerar el orden de las palabras, y para aprovecharla harían falta fraudes más sutiles que este dataset no permite medir. Decirlo demuestra comprensión.

**Resultado esperado**

```
Exactitud LSTM 0,967 | regresión logística 0,967
AUC LSTM 0,991 | regresión logística 0,995
Recall de fraude LSTM 0,945
(Los valores pueden variar entre ejecuciones y entre CPU y GPU.)
```

## Paso 14. Probar con correos nuevos · Escribir la función, pegar los correos · ~1,5 min

```python
# Paso 14: probar con correos nuevos
def clasificar_correo(texto, umbral=0.5):
    numeros = vectorizador([limpiar(texto)])
    prob = float(modelo.predict(numeros, verbose=0)[0, 0])
    return prob, ('FRAUDULENTO' if prob >= umbral else 'SEGURO')

correos = [
    ("Fraude", "Banco", """Subject: URGENT: Your Bank Account Will Be Suspended
Dear Customer, We have detected unusual activity on your bank account. For security reasons, we have temporarily restricted certain transactions.
To prevent your account from being permanently suspended, you must verify your personal information within the next 24 hours.
Please confirm your identity using the following link: https://secure-bank-validation.invalid/confirm
Failure to complete this verification may result in the suspension of your online banking access.
Sincerely, Bank Security Department"""),
    ("Fraude", "Paquete", """Subject: Delivery Failed: Action Required for Your Package
Hello, Your package has arrived at our distribution center, but we were unable to complete the delivery due to an issue with the registered address.
To schedule another delivery attempt, please confirm your shipping information and pay a small processing fee of $2.99.
You can update your delivery details here: https://parcel-delivery-update.invalid/reschedule
Your package will be held for 48 hours before being returned to the sender.
Thank you for your understanding. Express Delivery Support Team"""),
    ("Fraude", "Soporte TI", """From: Academic IT Support <support@campus-account-services.invalid>
Subject: Important Update Regarding Your Student Account
Dear Student, We are currently updating the authentication systems used by our academic platform.
As part of this process, some students are required to confirm their account information to maintain access to their courses and learning materials.
You can review your account status through the following portal: https://campus-account-services.invalid/verification
This procedure ensures that your student account remains correctly connected to our institutional services.
Best regards, Academic IT Support Team"""),
    ("Seguro", "Coordinación", """From: Academic Coordination <coordination@university-example.invalid>
Subject: Information About Upcoming Project Presentations
Dear Students, We would like to inform you that project presentations for this semester will begin next week.
Each group should prepare a presentation lasting approximately 10 minutes, covering the project objectives, technologies used, and main results achieved.
Please review the evaluation rubric available in your course materials to ensure that all required topics are included.
If you have any questions, feel free to ask during the next class.
Best regards, Academic Coordination Office"""),
]
for real, nombre, correo in correos:
    prob, decision = clasificar_correo(correo)
    print(f"{nombre:13s} | Real: {real:7s} | Modelo: {decision:12s} | P(fraude)={prob:6.1%}")
```

**Qué explicar**

- Escribo la función en vivo y pego los correos: lo que importa aquí es la función, no tipear los textos.
- `limpiar` y `vectorizador` son los mismos del entrenamiento: el correo nuevo pasa por el mismo proceso.
- Los correos 1 y 2 usan urgencia, amenazas o un cobro. El 4 es un aviso académico normal y debe salir seguro. Ojo: comparte el vocabulario académico del correo 3, así que por sí solo no prueba mucho.
- **El correo 3 es el importante:** es un fraude bien escrito, sin amenazas. Si el modelo lo deja pasar, lo digo: ese es su límite real.
- Muestro la probabilidad, no solo la etiqueta.

**Por qué así** (para responder preguntas)

- Esta es la prueba con datos que pide la rúbrica. Los correos están en inglés porque el dataset está en inglés.
- Probable explicación del correo 3: el vocabulario académico. En los datos limpios, 'university' aparece en el 14,7% de los correos seguros y en el 0,9% de los fraudes, y 'linguistics' en el 10,1% contra 0%: muchos correos seguros vienen de listas académicas. Además, el modelo no analiza el remitente ni el dominio real del enlace.
- La mejora concreta que se desprende de esto es analizar el remitente y el dominio real de los enlaces.

**Resultado esperado**

```
Banco        | Fraude | FRAUDULENTO | 99,1%
Paquete      | Fraude | FRAUDULENTO | 97,8%
Soporte TI   | Fraude | SEGURO      | 4,1%  (fraude que se le escapa)
Coordinación | Seguro | SEGURO      | 0,4%
(Los valores pueden variar entre ejecuciones.)
```

## Paso 15. Interpretar el resultado · Escribir · ~0,5 min

```python
# Paso 15: interpretar en términos del problema
vn, fp, fn, vp = confusion_matrix(y_prueba, pred_lstm).ravel()
print(f"De {vp+fn} correos fraudulentos, detectó {vp} ({vp/(vp+fn):.1%}) y dejó pasar {fn}.")
print(f"De {vn+fp} correos seguros, {fp} ({fp/(vn+fp):.1%}) quedarían marcados por error.")
```

**Qué explicar**

- Conecto con el problema: el modelo sirve como filtro de apoyo. Lo marcado va a cuarentena para revisión, no se borra.
- Los fraudes que deja pasar son el riesgo principal, así que no reemplaza la capacitación ni otros controles.

**Por qué así** (para responder preguntas)

- Este cierre responde la pregunta de la rúbrica sobre qué evidencia demuestra que el modelo contribuye a resolver el problema.

**Resultado esperado**

```
De 1289 correos fraudulentos, detectó 1218 (94.5%) y dejó pasar 71.
De 2132 correos seguros, 41 (1.9%) quedarían marcados por error.
```

## Paso 16 (opcional). Efecto del umbral · Escribir · fuera del tiempo

```python
# Paso 16: efecto del umbral
for umbral in [0.3, 0.5, 0.7, 0.9]:
    pred = (prob_lstm >= umbral).astype(int)
    vn_u, fp_u, fn_u, vp_u = confusion_matrix(y_prueba, pred).ravel()
    print(f"umbral {umbral}: detecta {vp_u/(vp_u+fn_u):.1%} del fraude | "
          f"falsas alarmas {fp_u/(vn_u+fp_u):.1%}")
```

**Qué explicar**

- El umbral depende de qué error le cuesta más a la organización.

**Por qué así** (para responder preguntas)

- Muestra criterio sobre el costo de cada tipo de error.

**Resultado esperado**

Cuatro líneas.

## Paso 17. Guardar el modelo (para la defensa) · Escribir · fuera del tiempo

```python
# Paso 17: guardar el modelo completo
modelo_completo = keras.Sequential([
    keras.Input(shape=(1,), dtype='string'),
    vectorizador,
    modelo
])
modelo_completo.save('modelo_phishing.keras')
print("Modelo guardado.")
try:
    from google.colab import files
    files.download('modelo_phishing.keras')
except ImportError:
    pass
```

**Qué explicar**

- Se puede ejecutar fuera de cámara.

**Por qué así** (para responder preguntas)

- Junta el vectorizador con la red, para que el archivo reciba texto directamente.

**Resultado esperado**

`Modelo guardado.` y la descarga del archivo.

---

## Cierre del video (30 a 60 segundos)

- **¿Resuelve el problema?** Parcialmente: sirve como filtro de apoyo contra el fraude evidente, pero dejó pasar el correo de soporte TI, que estaba bien escrito.
- **Comparación honesta:** la regresión logística rinde prácticamente igual en este dataset.
- **Limitaciones:** inglés, mezcla de phishing y spam, muchos correos seguros ligados a una sola empresa, fraudes bien escritos que se le escapan y falsas alarmas en correos automáticos.
- **Mejoras:** correos recientes y en español, señales del remitente y de los enlaces, y clases separadas para phishing y spam.

## Plan B si algo falla en vivo

| Problema | Qué hacer |
|---|---|
| Falla la descarga | Volver a ejecutar. Si persiste, bajar el CSV desde la página de Kaggle, subirlo al panel de archivos de Colab y leerlo con `pd.read_csv('Phishing_Email.csv')`. |
| `NameError` | Se saltó una celda: ejecutar las anteriores. |
| Error de tipeo | Leer el mensaje en voz alta y corregir. |
| Resultados distintos a los de esta guía | Es esperable: interpretar los números que salieron. |

## Modo defensa

Ejecutar los pasos 1, 2, 3, 5 y 6, subir `modelo_phishing.keras` al panel de archivos de Colab y ejecutar la celda **Modo defensa**.
