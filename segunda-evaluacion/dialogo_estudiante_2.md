# Diálogo del estudiante 2: programación en vivo

Lo que dice [Nombre estudiante 2] mientras escribe el código, bloque por bloque. Es el mismo código de `guia_phishing_en_vivo.md` y del notebook de respaldo. Las razones detrás de cada decisión, para las preguntas, están en esa guía y en `decisiones_del_proyecto.md`.

## Cómo usar este diálogo

- Cada paso está dividido en bloques chicos de código. Escribes el bloque mientras dices el texto de **Mientras escribes**. Después ejecutas y dices el texto de **Cuando sale el resultado**.
- **Todos los bloques de un paso van en la misma celda.** La división es solo para saber qué decir mientras escribes cada parte. Escribe el primer bloque, sigue con el siguiente en la misma celda y ejecuta una sola vez al final del paso (Shift + Enter). Es una celda por paso, como en `phishing_en_vivo.ipynb`.
- Lo que va en cursiva y entre paréntesis es una acción, no se dice.
- **Los números pueden cambiar un poco en Colab** (otra versión de TensorFlow y GPU). Di los que salgan en pantalla, no los de esta guía.
- Se puede pegar el paso 12 (gráficos) y los textos de los correos del paso 14. Todo lo demás se escribe.
- Tiempo máximo: 15:00. Los relojes de cada paso suman 14:15 y dejan 45 segundos para errores de tipeo. Ensaya completo en Colab con cronómetro: escribir en vivo siempre toma más de lo que parece.
- No leas: quédate con las ideas en **negrita** y dilo con tus palabras.

## Antes de grabar (Google Colab)

- [ ] Entrar a colab.research.google.com con una cuenta de Google y subir `phishing_en_vivo.ipynb` (*Archivo → Subir notebook*). Trae solo los títulos de cada paso y una celda vacía debajo de cada uno.
- [ ] Activar la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU → Guardar*. Si Colab no asigna GPU, el código funciona igual en CPU (en el ensayo, poco más de un minuto de entrenamiento).
- [ ] Ensayar completo y cronometrar en una **copia** del notebook (*Archivo → Guardar una copia en Drive*), para que el notebook del video siga vacío. El código se ensayó en un computador local y todavía no en Colab: este ensayo es obligatorio.
- [ ] Antes de grabar: *Entorno de ejecución → Desconectar y borrar el entorno de ejecución*, para partir de cero, y volver a activar la GPU si hace falta.
- [ ] Subir el zoom del navegador (Ctrl y +) para que el código se lea en el video. Cerrar notificaciones y otras pestañas.
- [ ] Tener el notebook de respaldo `deteccion_phishing_lstm.ipynb` abierto en otra pestaña.

---

## Paso 1 · Preparar el entorno · 0:00–0:35

```python
# Paso 1: preparar el entorno
%pip install -q kagglehub
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPU disponible:", tf.config.list_physical_devices('GPU'))
```

**Mientras escribes:**
Hola, soy [Nombre estudiante 2] y voy a construir la solución en vivo. **Estoy en Google Colab, con la GPU activada**: la GPU acelera los cálculos de la red, que se repiten por cada palabra del correo. Colab ya trae TensorFlow, scikit-learn y pandas. Con %pip me aseguro de tener kagglehub, que descarga el dataset directo desde Kaggle. Después importo TensorFlow y reviso la versión y si hay GPU.

**Cuando sale el resultado:**
Aquí está la versión de TensorFlow y la GPU que asignó Colab.
*(si la lista de GPU sale vacía)* Colab no asignó GPU, así que la red entrena en el procesador. Funciona igual, solo demora un poco más: en el ensayo, poco más de un minuto.

## Paso 2 · Importar las bibliotecas · 0:35–1:05

```python
# Paso 2: importar bibliotecas
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
```

**Mientras escribes:**
re sirve para limpiar texto con patrones. numpy y pandas manejan los datos como tabla, y matplotlib hace los gráficos.

```python
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, ConfusionMatrixDisplay)

keras.utils.set_random_seed(42)   # resultados repetibles
```

**Mientras escribes:**
De Keras uso las capas para armar la red. De scikit-learn uso tres cosas: separar los datos, el modelo simple de comparación y las métricas. Y **fijo una semilla, para que el experimento se pueda repetir**.

**Cuando sale el resultado:** no hay salida; sigo.

## Paso 3 · Cargar los datos · 1:05–1:35

```python
# Paso 3: cargar los datos
ruta = kagglehub.dataset_download('subhajournal/phishingemails')
datos = pd.read_csv(f'{ruta}/Phishing_Email.csv')
datos = datos.rename(columns={'Email Text': 'texto', 'Email Type': 'tipo'})[['texto', 'tipo']]
datos.head()
```

**Mientras escribes:**
Descargo el dataset Phishing Email Detection, de Kaggle, y lo leo como tabla. Me quedo con dos columnas: el texto del correo, que es la entrada, y el tipo, que es lo que quiero predecir. **Cada correo trae su respuesta correcta: por eso es aprendizaje supervisado.**

**Cuando sale el resultado:**
Estas son las primeras filas: el texto y su etiqueta, Safe Email o Phishing Email.

## Paso 4 · Explorar la estructura de los datos · 1:35–2:50

```python
# Paso 4: explorar la estructura de los datos
print("Filas:", len(datos), "| Texto vacío:", datos['texto'].isna().sum())
print("Duplicados exactos:", datos['texto'].duplicated().sum())
print(datos['tipo'].value_counts())
```

**Mientras escribes:**
Antes de entrenar, reviso los datos. Cuento las filas, los textos vacíos y los duplicados, y cuántos correos hay de cada tipo.

```python
palabras = datos['texto'].dropna().str.split().str.len()
print("Mediana de palabras por correo:", int(palabras.median()),
      "| 90% de los correos tiene menos de:", int(palabras.quantile(0.9)))
```

**Mientras escribes:**
Mido el largo de los correos en palabras. Me sirve para decidir cuántas palabras va a leer la red.

```python
con_enron = datos['texto'].fillna('').str.lower().str.contains(r'\benron\b')
print("\n% de correos que dicen 'enron', por tipo:")
print((con_enron.groupby(datos['tipo']).mean() * 100).round(1))
```

**Mientras escribes:**
Y reviso una palabra sospechosa: enron, el nombre de una empresa.

**Cuando sale el resultado:**
Hay 18.650 correos, 16 vacíos y 1.112 duplicados. **Los duplicados los voy a quitar**: un correo repetido podría quedar en entrenamiento y en prueba, y eso inflaría el resultado. Hay más seguros que fraudulentos, cerca de 6 de cada 10. No es un desbalance fuerte, pero por eso no me basta la exactitud: también voy a medir recall y precisión. La mediana es de 160 palabras.
*(señala la última tabla)* Y el hallazgo clave: **enron aparece en el 20% de los correos seguros y en ningún fraude**. Si lo dejo, el modelo aprende un atajo: si dice enron, es seguro. Eso no sirve en ninguna otra organización.

## Paso 5 · Limpiar el texto · 2:50–4:05

```python
# Paso 5: limpiar el texto
def limpiar(texto):
    texto = str(texto).lower()
```

**Mientras escribes:**
Escribo una función para limpiar cada correo. Primero, todo a minúsculas, para que "Urgent" y "urgent" sean la misma palabra.

```python
    texto = re.sub(r'<[^>]+>', ' ', texto)                           # etiquetas HTML
    texto = re.sub(r'(https?://|www\.)\S+', ' enlaceweb ', texto)    # enlaces
```

**Mientras escribes:**
re.sub busca un patrón y lo reemplaza. Quito las etiquetas HTML, que son formato y no contenido. Y **cambio cada enlace por la palabra enlaceweb: me importa que haya un enlace, no cuál es**.

```python
    texto = re.sub(r'\b(enron|vince|kaminski|hou|ect)\b', ' ', texto)  # atajos del dataset
    texto = re.sub(r'[^a-záéíóúñü ]', ' ', texto)                    # solo letras
    return re.sub(r'\s+', ' ', texto).strip()
```

**Mientras escribes:**
Aquí **borro el atajo**: enron y otras palabras que delatan a esa empresa, como nombres y partes de sus direcciones. La barra b es borde de palabra: así no borro "ect" dentro de "project". Después dejo solo letras, con tildes y ñ, y junto los espacios repetidos.

```python
datos = datos.dropna(subset=['texto'])
datos['limpio'] = datos['texto'].map(limpiar)
datos = datos[datos['limpio'] != ''].drop_duplicates('limpio')
print("Correos únicos:", len(datos))
print(datos['tipo'].value_counts())
print("\nEjemplo:", datos['limpio'].iloc[0][:150])
```

**Mientras escribes:**
Aplico la función a todos los correos. Quito los vacíos y **elimino duplicados después de limpiar**, porque dos correos pueden quedar iguales recién al limpiarlos.

**Cuando sale el resultado:**
Quedan 17.103 correos únicos: 10.661 seguros y 6.442 fraudulentos. Así se ve un correo limpio. **Esta misma función la voy a usar con los correos nuevos del final**: el modelo tiene que recibir siempre el mismo formato.

## Paso 6 · Separar los datos · 4:05–4:55

```python
# Paso 6: separar entrenamiento, validación y prueba
X = datos['limpio'].values
y = (datos['tipo'] == 'Phishing Email').astype(int).values   # 1 = fraudulento, 0 = seguro
```

**Mientras escribes:**
Separo la entrada de la etiqueta. La etiqueta pasa a número, 1 fraudulento y 0 seguro, porque la red trabaja con números.

```python
X_resto, X_prueba, y_resto, y_prueba = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
X_ent, X_val, y_ent, y_val = train_test_split(
    X_resto, y_resto, test_size=0.15, stratify=y_resto, random_state=42)
print("Entrenamiento:", len(X_ent), "| Validación:", len(X_val), "| Prueba:", len(X_prueba))
```

**Mientras escribes:**
Primero aparto el 20% para la prueba final. Del resto, aparto un 15% para validación. Con stratify, cada parte mantiene la misma proporción de fraudes.

**Cuando sale el resultado:**
Quedan 11.629 para entrenar, 2.053 para validar y 3.421 para la prueba. **Con el entrenamiento la red aprende. La validación vigila el sobreajuste. Y la prueba son correos que la red no ve hasta el final.**

## Paso 7 · Convertir las palabras en números · 4:55–5:50

```python
# Paso 7: convertir las palabras en números
VOCABULARIO, LARGO = 20000, 200
vectorizador = layers.TextVectorization(
    max_tokens=VOCABULARIO, output_sequence_length=LARGO, standardize=None)
vectorizador.adapt(X_ent)   # aprende el vocabulario solo con entrenamiento
```

**Mientras escribes:**
**La red no entiende texto, solo números.** Defino un vocabulario de 20.000 palabras y un largo de 200 por correo. TextVectorization reemplaza cada palabra por su número. standardize va en None porque ya limpié el texto. Y **adapt arma el vocabulario solo con los correos de entrenamiento**: si usara la prueba, le estaría filtrando información al modelo.

```python
X_ent_num = vectorizador(X_ent)
X_val_num = vectorizador(X_val)
X_prueba_num = vectorizador(X_prueba)
print("Forma:", X_ent_num.shape)
print("Correo:", X_ent[0][:60])
print("Números:", X_ent_num[0][:10].numpy())
```

**Mientras escribes:**
Convierto los tres grupos y muestro un ejemplo.

**Cuando sale el resultado:**
Cada correo quedó como 200 números. *(señala el ejemplo)* Workshop aparece dos veces, y las dos veces es el 149. Los correos cortos se rellenan con ceros, y los largos se cortan en la palabra 200. Es el mismo ejemplo que mostró [Nombre estudiante 1].

## Paso 8 · Crear una línea de comparación · 5:50–6:45

```python
# Paso 8: línea de comparación (regresión logística)
tfidf = TfidfVectorizer(max_features=VOCABULARIO)
base = LogisticRegression(max_iter=1000).fit(tfidf.fit_transform(X_ent), y_ent)
```

**Mientras escribes:**
Antes de la red, entreno un modelo simple: una regresión logística. **Un modelo complejo solo se justifica si aporta frente a uno simple.** TF-IDF da más peso a las palabras que distinguen a cada correo. Este modelo solo mira qué palabras aparecen, no su orden.

```python
prob_base = base.predict_proba(tfidf.transform(X_prueba))[:, 1]
print(classification_report(y_prueba, (prob_base >= 0.5).astype(int),
                            target_names=['Seguro', 'Fraudulento'], digits=3))
```

**Mientras escribes:**
Calculo la probabilidad de fraude en la prueba y muestro el reporte.

**Cuando sale el resultado:**
El modelo simple ya logra una exactitud de 0,967 y detecta el 94% de los fraudes. **Esa es la vara que la LSTM tiene que superar.**

## Paso 9 · Construir la arquitectura LSTM · 6:45–7:55

```python
# Paso 9: construir la arquitectura
modelo = keras.Sequential([
    keras.Input(shape=(LARGO,)),
    layers.Embedding(VOCABULARIO, 64, mask_zero=True),  # ignora el relleno de ceros
```

**Mientras escribes:**
Ahora la red. Es un modelo secuencial: las capas van una tras otra. Entran los 200 números del correo. La primera capa es el Embedding: **convierte cada número en un vector de 64 valores, que se aprenden al entrenar**. Con mask_zero le digo que los ceros son relleno, no palabras.

```python
    layers.LSTM(64),
    layers.Dropout(0.5),
```

**Mientras escribes:**
Después, la LSTM con 64 unidades. **Lee el correo palabra por palabra, con memoria: por eso la elegimos, porque el orden importa.** El Dropout, solo al entrenar, pone en cero al azar la mitad de los valores que entrega la LSTM. Así la red no depende de unas pocas unidades y se sobreajusta menos.

```python
    layers.Dense(1, activation='sigmoid')
])
modelo.summary()
```

**Mientras escribes:**
Y al final, una neurona con sigmoide, que entrega la probabilidad de fraude, entre 0 y 1. Veo el resumen.

**Cuando sale el resultado:**
Son 1.313.089 parámetros, y casi todos están en el Embedding: 20.000 palabras por 64 valores. Es la arquitectura que explicó [Nombre estudiante 1].
*(si aparece un aviso sobre mask)* Este aviso no es un error: la LSTM sí usa la máscara. Lo comprobamos en el ensayo.

## Paso 10 · Configurar el entrenamiento · 7:55–8:25

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

**Mientras escribes:**
Configuro el entrenamiento. **La pérdida es la entropía cruzada binaria**, la que corresponde a una salida sigmoide. El optimizador es Adam: usa el gradiente del error, que se calcula con retropropagación, para ajustar cada peso. Además de la exactitud, mido **recall, porque un fraude que se escapa puede terminar en una clave robada**, y precisión, porque una falsa alarma puede esconder un correo legítimo.

**Cuando sale el resultado:** no hay salida; sigo.

## Paso 11 · Entrenar el modelo · 8:25–10:05

```python
# Paso 11: entrenar el modelo
parada = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=2, restore_best_weights=True)
```

**Mientras escribes:**
Defino una parada temprana. Vigila la pérdida de validación: **si no mejora en 2 épocas, detiene el entrenamiento y recupera los mejores pesos.**

```python
historial = modelo.fit(
    X_ent_num, y_ent,
    validation_data=(X_val_num, y_val),
    epochs=6,
    batch_size=32,
    callbacks=[parada],
    verbose=2
)
```

**Mientras escribes:**
Entreno con los correos de entrenamiento y valido con los de validación. Máximo 6 épocas, en lotes de 32 correos.

**Mientras entrena** *(con GPU puede tardar solo unos segundos por época; si termina antes, completa la idea mirando las líneas de cada época)*:
Mientras entrena, explico qué pasa. La red toma 32 correos, calcula su probabilidad y la compara con la etiqueta real usando la pérdida. Luego **la retropropagación calcula cuánto aportó cada peso al error, y Adam ajusta los pesos un poco**. Eso se repite 364 veces por época. *(señala una línea)* loss es la pérdida en entrenamiento. val_loss es la pérdida en correos que no se usan para ajustar pesos. **Si loss baja y val_loss sube, la red está memorizando: eso es sobreajuste**, y ahí actúa la parada temprana.

**Cuando termina:**
Se detuvo en la época *(di la que salga)* y recuperó los pesos de la mejor época. En el ensayo se detuvo en la 4 y se quedó con la 2.

## Paso 12 · Curvas de aprendizaje · 10:05–10:30 · se puede pegar

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

**Mientras pegas:**
Este código solo dibuja los gráficos, así que lo pego.

**Cuando sale el resultado:**
A la izquierda está la pérdida; a la derecha, la exactitud. La pérdida de entrenamiento tiende a bajar, aunque no en todas las épocas. Miro la de validación: la más baja fue en la época *(di cuál)*, y la parada temprana se quedó con esos pesos. **Si la validación sube mientras el entrenamiento baja, la red está memorizando: eso es sobreajuste.**
*(si la curva de validación no sube)* Aquí la validación no se separó del entrenamiento, así que no hubo un sobreajuste claro.

## Paso 13 · Evaluar y comparar · 10:30–11:35

```python
# Paso 13: evaluar y comparar
prob_lstm = modelo.predict(X_prueba_num, verbose=0).ravel()
pred_lstm = (prob_lstm >= 0.5).astype(int)
print(classification_report(y_prueba, pred_lstm,
                            target_names=['Seguro', 'Fraudulento'], digits=3))
```

**Mientras escribes:**
Ahora evalúo con la prueba: 3.421 correos que la red nunca vio. Calculo la probabilidad de fraude y uso 0,5 como umbral.

```python
print("AUC LSTM:", round(roc_auc_score(y_prueba, prob_lstm), 3),
      "| AUC regresión logística:", round(roc_auc_score(y_prueba, prob_base), 3))
print("Exactitud LSTM:", round((pred_lstm == y_prueba).mean(), 3),
      "| regresión logística:", round(((prob_base >= 0.5).astype(int) == y_prueba).mean(), 3))
ConfusionMatrixDisplay(confusion_matrix(y_prueba, pred_lstm),
                       display_labels=['Seguro', 'Fraudulento']).plot(cmap='Blues')
plt.show()
```

**Mientras escribes:**
Comparo el AUC y la exactitud con la regresión logística, y dibujo la matriz de confusión.

**Cuando sale el resultado:**
La LSTM logra una exactitud de 0,967, un recall de fraude de 0,945 y un AUC de 0,991. *(señala la matriz)* La diagonal son aciertos. Abajo a la izquierda están los fraudes que se escaparon, y arriba a la derecha, las falsas alarmas.
Y lo digo sin rodeos: **en este dataset la LSTM rinde prácticamente igual que la regresión logística.** Empatan en exactitud, y la regresión tiene un AUC un poco mayor. Con estos datos, las palabras por sí solas ya separan muy bien los correos.

## Paso 14 · Probar con correos nuevos · 11:35–12:55 · los correos se pegan

```python
# Paso 14: probar con correos nuevos
def clasificar_correo(texto, umbral=0.5):
    numeros = vectorizador([limpiar(texto)])
    prob = float(modelo.predict(numeros, verbose=0)[0, 0])
    return prob, ('FRAUDULENTO' if prob >= umbral else 'SEGURO')
```

**Mientras escribes:**
Ahora la prueba con correos nuevos. Escribo una función que recibe un texto, **lo limpia con la misma función de antes y lo convierte con el mismo vectorizador**. Después le pide la probabilidad a la red. Si es 0,5 o más, dice fraudulento.

```python
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
```

**Mientras pegas:**
Los correos los pego, porque lo importante es la función. Son cuatro, escritos para esta prueba: el del banco que mostró [Nombre estudiante 1], uno de un paquete con un cobro, uno de soporte TI y un aviso académico normal. Los enlaces usan el dominio .invalid, que no existe.

```python
for real, nombre, correo in correos:
    prob, decision = clasificar_correo(correo)
    print(f"{nombre:13s} | Real: {real:7s} | Modelo: {decision:12s} | P(fraude)={prob:6.1%}")
```

**Mientras escribes:**
Recorro los cuatro y muestro la probabilidad, no solo la etiqueta.

**Cuando sale el resultado:**
El del banco sale fraudulento, con 99%: urgencia, amenaza y enlace. El del paquete también, con 98%. El aviso académico sale seguro, como corresponde.
*(señala la línea de soporte TI)* Y este es el importante. **Es un fraude bien escrito, sin amenazas ni urgencia, y el modelo lo deja pasar**, con 4%. Ese es su límite real. **Probablemente se debe al vocabulario académico**: en este dataset, palabras como university o students aparecen casi solo en correos seguros. Además, el modelo no analiza el remitente ni el dominio real del enlace.

## Paso 15 · Interpretar el resultado · 12:55–13:25

```python
# Paso 15: interpretar en términos del problema
vn, fp, fn, vp = confusion_matrix(y_prueba, pred_lstm).ravel()
print(f"De {vp+fn} correos fraudulentos, detectó {vp} ({vp/(vp+fn):.1%}) y dejó pasar {fn}.")
print(f"De {vn+fp} correos seguros, {fp} ({fp/(vn+fp):.1%}) quedarían marcados por error.")
```

**Mientras escribes:**
Por último, traduzco las métricas al problema que presentó [Nombre estudiante 1].

**Cuando sale el resultado:**
De 1.289 correos fraudulentos, detectó 1.218, el 94,5%, y dejó pasar 71. De 2.132 correos seguros, 41 quedarían marcados por error, menos del 2%. **Para el área de TI, el modelo sirve como filtro de apoyo: lo marcado va a cuarentena para revisarlo, no se borra.** Y como algunos fraudes se escapan, no reemplaza la capacitación ni otros controles.

## Cierre del video · 13:25–14:15

*(sin código, mirando a cámara)*
Entonces, ¿resuelve el problema? **Parcialmente.** Sirve como filtro de apoyo contra el fraude evidente, pero dejó pasar un fraude bien escrito. Además, la regresión logística rindió prácticamente igual en este dataset, y lo mostramos tal cual. Las limitaciones son claras: los datos están en inglés, mezclan phishing con spam y muchos correos seguros vienen de una sola empresa. **Las mejoras: entrenar con correos recientes y en español, sumar el remitente y el dominio real de los enlaces, y separar el phishing del spam.**

---

## Fuera de cámara

**Paso 17: guardar el modelo para la defensa.** Ejecútalo apenas termines de grabar: descarga `modelo_phishing.keras` a tu computador (si el navegador pregunta, permite la descarga). Sin ese archivo, en la defensa habría que volver a entrenar. El código está en `guia_phishing_en_vivo.md`. Si te preguntan: el archivo junta el vectorizador con la red, así que en la defensa recibe texto directamente, sin volver a entrenar.

**Paso 16 (opcional), el umbral.** Si sobra tiempo o lo preguntan en la defensa: con un umbral de 0,3 detecta más fraude, pero con más falsas alarmas; con 0,9 casi no hay falsas alarmas, pero se escapan más fraudes. **El umbral lo decide la organización según qué error le cuesta más.**

## Si algo falla en vivo

| Problema | Qué decir y hacer |
|---|---|
| Error de tipeo | "Me equivoqué en esta línea, la corrijo." Lee el mensaje de error en voz alta y corrige: también demuestra que entiendes el código. |
| `NameError` | "Me salté una celda." *Entorno de ejecución → Ejecutar anteriores* (Ctrl + F8). |
| Falla la descarga | Vuelve a ejecutar. Si sigue fallando, descarga el CSV desde la página de Kaggle, súbelo al panel de archivos de Colab (ícono de carpeta, a la izquierda) y léelo con `pd.read_csv('Phishing_Email.csv')`. |
| Colab se desconecta o se queda pegado | "Se reinició el entorno." *Entorno de ejecución → Reiniciar sesión* y después *Ejecutar anteriores*. |
| Colab no asigna GPU | "Hoy no tengo GPU, así que entreno en el procesador." Sigue igual: tarda cerca de un minuto. |
| Números distintos a esta guía | "Los números varían un poco entre ejecuciones." Interpreta los que salieron. |

## Si me quedo en blanco

- **Paso 1:** me presento · Colab con GPU · %pip kagglehub · versión y GPU
- **Paso 2:** para qué sirve cada biblioteca · semilla
- **Paso 3:** dataset de Kaggle · texto y tipo · supervisado
- **Paso 4:** duplicados inflan · 6 de cada 10 seguros · enron es un atajo
- **Paso 5:** minúsculas · HTML · enlaceweb · borro el atajo · duplicados después
- **Paso 6:** 1 y 0 · entrenamiento, validación, prueba · stratify
- **Paso 7:** solo números · 20.000 y 200 · adapt solo con entrenamiento · el 149
- **Paso 8:** modelo simple primero · TF-IDF · no ve el orden · la vara
- **Paso 9:** Embedding 64 · mask · LSTM con memoria · Dropout · sigmoide · 1,3 millones
- **Paso 10:** entropía cruzada binaria · Adam · recall y precisión
- **Paso 11:** parada temprana · lotes de 32 · retropropagación · loss y val_loss
- **Paso 12:** validación sube = memoriza
- **Paso 13:** 0,967 · diagonal · empate con la regresión
- **Paso 14:** misma limpieza · banco y paquete fraude · aviso seguro · soporte TI se escapa · vocabulario académico
- **Paso 15:** 1.218 de 1.289 · 41 falsas alarmas · cuarentena, no borrar
- **Cierre:** parcialmente · empate honesto · limitaciones · mejoras
