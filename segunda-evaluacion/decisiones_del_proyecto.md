# Por qué hicimos cada cosa: decisiones del proyecto

Proyecto: detección de correos fraudulentos (phishing) con una red LSTM en Keras. Grupo: [Nombre estudiante 1] (problema y modelo) y [Nombre estudiante 2] (programación en vivo).

## Cómo usar esta guía

- Es material para preparar la defensa: 5 minutos por integrante con una infografía individual, más preguntas sobre la parte propia y sobre la solución completa. No es un texto para leer en voz alta: la rúbrica castiga la lectura, así que conviene estudiarla y responder con palabras propias.
- Cada decisión tiene el mismo formato: qué hicimos, por qué, alternativas que no usamos, qué decir si preguntan y la evidencia. Las etiquetas significan: **[Medido en el notebook]** = sale de una salida del notebook ejecutado; **[Comprobado en ensayo]** = se verificó en el ensayo local, fuera del notebook; **[Criterio técnico]** = razonamiento técnico, no medido; **[Requisito del enunciado]** = lo exige el enunciado.
- Las cifras salen del notebook de respaldo `deteccion_phishing_lstm.ipynb`, ejecutado en un computador local con CPU. En la grabación pueden variar un poco: si cambian, se usan las que salgan en vivo. Lo marcado como "inferencia" o "interpretación" es razonamiento nuestro, no una medición.

## Índice

1. Planificación y roles
2. El problema
3. Los datos
4. Exploración
5. Limpieza
6. Separación
7. Vectorización
8. Línea base
9. Arquitectura
10. Compilación
11. Entrenamiento
12. Curvas de aprendizaje
13. Evaluación
14. Prueba con correos nuevos
15. Interpretación
16. Umbral
17. Guardar el modelo completo y modo defensa
18. Semilla y reproducibilidad
19. Conclusión: "parcialmente"
20. Ética y ley
21. La pregunta difícil: ¿por qué Deep Learning si la regresión logística empata?
22. Cifras para tener a mano
23. Limitaciones que reconocemos
24. Mejoras
25. Comparación con el material del profesor

---

## 1. Planificación y roles

### Dos estudiantes para tres roles

**Qué hicimos:** El enunciado define tres roles: integrante 1 (problema), integrante 2 (modelo y marco teórico) e integrante 3 (programación en vivo). Somos dos. [Nombre estudiante 1] cubre los roles 1 y 2 con una presentación de unos 13,5 minutos (diapositivas 1 a 15). [Nombre estudiante 2] cubre el rol 3 y programa en vivo en máximo 15 minutos. El video cierra diciendo que la solución responde parcialmente al problema y cuáles son las mejoras principales.

**Por qué:** El enunciado permite grupos de máximo 3 personas y fija un máximo de 15 minutos por integrante. La programación en vivo ya ocupa unos 15,0 minutos estimados en la guía en vivo, así que a quien programa no le cabe otro rol. El problema y el modelo son un mismo argumento (el problema justifica el modelo), y el enunciado valora especialmente esa coherencia: conviene que lo cuente una sola persona de corrido. El enunciado no dice cómo repartir los roles en un grupo de dos: este reparto es una decisión nuestra.

**Alternativas que no usamos:** Repartir la teoría entre los dos: [Nombre estudiante 2] pasaría de los 15 minutos, porque la programación en vivo ya usa todo su tiempo.

**Si preguntan:** "Como somos dos, uno tomó el problema y el modelo, que son un mismo argumento, y el otro la implementación, que por sí sola ocupa los 15 minutos."

**Evidencia:** [Requisito del enunciado]

### [Nombre estudiante 1] no adelanta resultados

**Qué hicimos:** La presentación de [Nombre estudiante 1] explica el problema, los datos y el modelo, pero no muestra exactitud, AUC, cuál modelo rinde mejor, el resultado de los 4 correos de prueba, la tabla de umbrales ni las curvas reales. Todo eso lo muestra [Nombre estudiante 2] en vivo.

**Por qué:** El enunciado pide la secuencia problema → modelo → implementación → resultados, y le pide al integrante 3 mostrar la evidencia. Además, los números pueden variar un poco entre ejecuciones y entre CPU y GPU: si se anunciaran antes, podrían no coincidir con los que salgan en la grabación.

**Alternativas que no usamos:** Poner las cifras del notebook de respaldo en las diapositivas: se arriesga una contradicción con la ejecución en vivo.

**Si preguntan:** "Los resultados son evidencia de la implementación, por eso los muestra quien la ejecuta, con los números que salieron en vivo."

**Evidencia:** [Requisito del enunciado]

### Programar en vivo con una guía y un notebook de respaldo

**Qué hicimos:** [Nombre estudiante 2] escribe el código en un notebook nuevo y vacío, siguiendo `guia_phishing_en_vivo.md`. Los pasos fundamentales se escriben en vivo; solo se pegan el código de los gráficos (paso 12) y los textos de los 4 correos (paso 14). `deteccion_phishing_lstm.ipynb` tiene el mismo código, ya ejecutado, y queda de respaldo. Los pasos 16 y 17 quedan fuera del tiempo del video.

**Por qué:** El enunciado no permite presentar un programa terminado y limitarse a ejecutarlo: hay que construir los componentes fundamentales en vivo mientras se explican. La guía asegura que el código funcione y que cada paso tenga su explicación. El respaldo sirve si algo falla, como referencia de las cifras y para la defensa. El enunciado también exige entender cualquier fragmento sacado de documentación o de asistentes de IA: por eso cada paso de la guía trae "Qué explicar" y "Por qué así".

**Alternativas que no usamos:** Ejecutar el notebook terminado (lo prohíbe el enunciado). Escribir todo sin guía (más riesgo de errores y de pasarse del tiempo).

**Si preguntan:** "El código se escribió en vivo. El notebook de respaldo tiene lo mismo, y lo usamos para ensayar y para la defensa."

**Evidencia:** [Requisito del enunciado]

### Google Colab con GPU T4

**Qué hicimos:** El video se graba en Google Colab con GPU T4 (*Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*). El paso 1 se asegura de tener `kagglehub` (con `%pip install -q`, que instala en el mismo entorno del notebook y oculta el detalle) e imprime la versión de TensorFlow y si hay GPU.

**Por qué:** Colab ya trae TensorFlow, scikit-learn, pandas y matplotlib, así que no hay que instalar nada en el computador de cada uno, y la GPU acelera la LSTM, que repite sus cálculos palabra por palabra. Imprimir la versión y la GPU deja evidencia del entorno al inicio del video. Sin GPU el código funciona igual: en el ensayo local en CPU (TensorFlow 2.22.0-rc0) el entrenamiento tomó poco más de 1 minuto.

**Alternativas que no usamos:** Un computador local con VS Code: es donde se ensayó, pero obliga a instalar todo y depende de la máquina de cada uno.

**Si preguntan:** "Colab trae todo instalado y la GPU acelera la LSTM, pero no es obligatoria: en CPU entrenó en poco más de un minuto."

**Evidencia:** [Comprobado en ensayo] Ojo: el ensayo fue local; todavía falta ejecutarlo y cronometrarlo en Colab antes de grabar.

### Bibliotecas y estructura `Sequential`

**Qué hicimos:** pandas maneja la tabla, `re` limpia el texto, Keras construye la red, scikit-learn separa los datos, crea la regresión logística y calcula las métricas, y matplotlib dibuja los gráficos. `numpy` se importa, pero el código no lo llama directamente.

**Por qué:** Son bibliotecas estándar de Python para datos y Deep Learning. La estructura `Sequential` de Keras es la de los ejemplos del curso, así que el código se parece a lo visto en clases.

**Alternativas que no usamos:** Otro framework, como PyTorch: los ejemplos del curso usan Keras, y PyTorch exige escribir a mano el ciclo de entrenamiento.

**Si preguntan:** "Usé Keras porque su estructura Sequential es la de los ejemplos del curso, y scikit-learn para lo clásico: separar los datos, la línea base y las métricas."

**Evidencia:** [Criterio técnico]

---

## 2. El problema

### Detectar phishing en el correo de una organización

**Qué hicimos:** El área de TI de una empresa recibe correos que imitan a bancos o servicios para robar claves y datos, además de spam. Queremos clasificar automáticamente cada correo entrante como seguro o fraudulento a partir de su texto, y estimar la probabilidad de que sea fraude. La organización también puede ser una universidad: uno de los correos de prueba imita un aviso de cuenta estudiantil.

**Por qué:** El correo es una de las puertas de entrada más comunes de los ataques. Revisarlo a mano no escala, y basta con que una persona caiga para comprometer una cuenta. El enunciado advierte que no basta con decir "desarrollaremos un sistema de ciberseguridad": hay que precisar la situación, sus causas, consecuencias y contexto. Por eso acotamos el problema a una decisión clara (seguro o fraudulento), con datos disponibles para estudiarla. Así respondemos a los criterios 2.1.2 y 2.1.3: una aplicación coherente con una necesidad de la organización y elegida según esa necesidad.

**Alternativas que no usamos:** Un tema amplio como "ciberseguridad con IA": el enunciado lo pone justamente como ejemplo de lo que no basta.

**Si preguntan:** "No es ciberseguridad en general: es decidir, correo por correo, si es seguro o fraudulento, para que TI revise lo sospechoso antes de que alguien caiga."

**Evidencia:** [Requisito del enunciado]

### Un filtro de apoyo con cuarentena, no un bloqueo automático

**Qué hicimos:** El modelo entrega una probabilidad. Lo que supera el umbral va a cuarentena para que una persona lo revise; no se borra ni se bloquea automáticamente.

**Por qué:** El modelo se equivoca en los dos sentidos: en la prueba dejó pasar 71 fraudes y marcó 41 correos seguros. Un bloqueo automático escondería esos correos legítimos; con cuarentena y revisión humana, una falsa alarma cuesta un poco de tiempo, no la pérdida de un correo. Y como también se le escapan fraudes, el filtro no reemplaza la capacitación ni otros controles.

**Alternativas que no usamos:** Bloquear o borrar automáticamente. Reemplazar la revisión humana.

**Si preguntan:** "Es un filtro de apoyo: prioriza qué revisar. La decisión final sobre un correo dudoso la toma una persona."

**Evidencia:** [Medido en el notebook] Pasos 13 y 15. La cuarentena con revisión humana está en el planteamiento y en las mejoras del notebook.

---

## 3. Los datos

### Dataset "Phishing Email Detection" de Kaggle

**Qué hicimos:** Usamos el dataset `subhajournal/phishingemails` (archivo `Phishing_Email.csv`, unos 52 MB), descargado con `kagglehub.dataset_download`. Renombramos `Email Text` → `texto` y `Email Type` → `tipo`, y nos quedamos solo con esas dos columnas. Con `datos.head()` se muestran las primeras 5 filas, que sirven para explicar la estructura de los datos, como pide el enunciado.

**Por qué:** Es público, viene etiquetado (lo que exige el aprendizaje supervisado) y tiene 18.650 correos. En el ensayo se descargó sin cuenta de Kaggle, algo clave para programar en vivo sin credenciales, y su tamaño (52 MB) no es problema para un computador normal. Renombrar en español deja el código alineado con la explicación, y quedarse con dos columnas deja solo la entrada (texto) y lo que se quiere predecir (tipo).

**Alternativas que no usamos:** Correos reales de una empresa o de la universidad: no están etiquetados y contienen datos personales. Un dataset en español: no lo usamos en este proyecto y queda como mejora.

**Si preguntan:** "Lo elegimos porque es público, viene etiquetado y se descarga en vivo sin cuenta. Conocemos sus límites y los tratamos."

**Evidencia:** [Comprobado en ensayo] Plan B si falla la descarga: bajar el CSV desde la página de Kaggle, subirlo al panel de archivos de Colab y leerlo con `pd.read_csv('Phishing_Email.csv')`.

### Usarlo a pesar de sus limitaciones

**Qué hicimos:** Lo usamos sabiendo que está en inglés, que mezcla phishing con spam común y que cerca del 20% de los correos seguros menciona a una sola empresa (Enron).

**Por qué:** Las limitaciones se pueden medir y tratar. El atajo de Enron se midió en el paso 4 y se quitó en el paso 5; las demás quedan declaradas en la conclusión y se convierten en mejoras. Como el dataset mezcla phishing con spam, hablamos de "correos fraudulentos y spam", no solo de phishing.

**Alternativas que no usamos:** Ocultar las limitaciones: una defensa honesta es más sólida. Separar phishing de spam: el dataset trae solo dos etiquetas (`Phishing Email` y `Safe Email`), así que no se puede sin volver a etiquetar.

**Si preguntan:** "Sabemos que está en inglés, que mezcla spam y que tiene un sesgo hacia Enron. Lo medimos, quitamos el atajo de Enron y dejamos el resto como limitación declarada."

**Evidencia:** [Medido en el notebook] Paso 4: `enron` aparece en el 20,3% de los seguros y en el 0,0% de los fraudulentos.

### Etiquetas y aprendizaje supervisado (criterio 2.1.1)

**Qué hicimos:** Cada correo trae su etiqueta, y el modelo aprende la relación texto → etiqueta. En el paso 6, `y = 1` es fraudulento (`Phishing Email`) e `y = 0` es seguro (`Safe Email`).

**Por qué:** Tenemos datos etiquetados y queremos predecir una etiqueta conocida: eso es aprendizaje supervisado. La red trabaja con números, por eso la etiqueta se convierte a 0 y 1. El fraude es la clase 1 porque es lo que queremos detectar: así el recall y la precisión de la clase positiva hablan del fraude.

**Alternativas que no usamos:** Aprendizaje no supervisado, por ejemplo agrupar correos parecidos sin etiquetas: no diría qué grupo es fraude, y habría que etiquetar después.

**Si preguntan:** "Es supervisado: cada correo trae la respuesta correcta y la red aprende de esos ejemplos. No supervisado sería agrupar correos sin etiquetas."

**Evidencia:** [Requisito del enunciado] Criterio 2.1.1: reconocer el método de aprendizaje según el origen de la información.

---

## 4. Exploración

Todas estas revisiones se hacen en el paso 4, sobre los datos tal como vienen, antes de limpiar.

### Contar filas y textos vacíos

**Qué hicimos:** `len(datos)` y `isna().sum()`: 18.650 filas y 16 textos vacíos.

**Por qué:** Un correo sin texto no se puede clasificar. Además, `limpiar()` convierte su entrada con `str()`, y un vacío (NaN) se volvería la palabra "nan". Por eso en el paso 5 se eliminan con `dropna` antes de limpiar.

**Alternativas que no usamos:** Rellenar los vacíos con algún texto: sería inventar datos.

**Si preguntan:** "Eran 16 de 18.650. Sin texto no hay nada que clasificar, y además se habrían convertido en la palabra 'nan'."

**Evidencia:** [Medido en el notebook] Paso 4.

### Buscar duplicados exactos

**Qué hicimos:** `duplicated().sum()`: 1.112 duplicados exactos.

**Por qué:** Un correo repetido podría quedar a la vez en entrenamiento y en prueba. El modelo lo "reconocería" en vez de generalizar, y el resultado se inflaría. Los duplicados también dan más peso a los correos que se repiten.

**Alternativas que no usamos:** Dejarlos: el resultado de la prueba dejaría de medir correos realmente nuevos.

**Si preguntan:** "Si el mismo correo está en entrenamiento y en prueba, la prueba ya no mide correos nuevos."

**Evidencia:** [Medido en el notebook] Paso 4.

### Revisar el balance de clases

**Qué hicimos:** `value_counts()`: 11.322 seguros y 7.328 fraudulentos (cerca de 61% y 39%).

**Por qué:** Con clases desbalanceadas, la exactitud sola engaña: en la prueba, responder siempre "seguro" ya acertaría cerca del 62% (2.132 de 3.421) sin detectar ningún fraude. El desbalance es moderado, así que no rebalanceamos los datos, pero sí separamos con `stratify` y medimos recall y precisión.

**Alternativas que no usamos:** Rebalancear (quitar correos seguros, repetir fraudes o dar pesos por clase): con un desbalance moderado no hacía falta alterar los datos; bastaba con medir con las métricas adecuadas. No lo probamos.

**Si preguntan:** "Hay más seguros que fraudes, pero no es extremo. Por eso no rebalanceé, y sí medí recall y precisión, no solo exactitud."

**Evidencia:** [Medido en el notebook] Pasos 4 y 13.

### Medir el largo de los correos

**Qué hicimos:** Mediana de 160 palabras; el 90% de los correos tiene menos de 729. Se cuenta separando por espacios el texto original, antes de limpiar.

**Por qué:** Para elegir cuántas palabras lee la red (`LARGO = 200`, paso 7) mirando los datos y no al azar.

**Alternativas que no usamos:** Elegir el largo sin mirar los datos.

**Si preguntan:** "Lo medí para decidir el largo de entrada: 200 palabras cubren más que la mediana, que es 160."

**Evidencia:** [Medido en el notebook] Paso 4.

### Medir la palabra "enron" por clase

**Qué hicimos:** Porcentaje de correos que contienen la palabra `enron` (con `\b`, como palabra completa), separado por tipo: 20,3% de los seguros y 0,0% de los fraudulentos.

**Por qué:** Enron es el nombre de una empresa y delata de dónde vienen muchos correos seguros. Si se deja, el modelo aprende un atajo: "dice enron, entonces es seguro". Ese atajo funciona en este dataset y no sirve en ninguna otra organización. Por eso se revisa antes de entrenar.

**Alternativas que no usamos:** No revisarlo: el modelo podría verse bien aquí por una razón que no sirve afuera.

**Si preguntan:** "Buscamos atajos antes de entrenar. 'enron' aparece en el 20% de los seguros y prácticamente en ningún fraude: es una pista del origen del dato, no del fraude."

**Evidencia:** [Medido en el notebook] Paso 4.

---

## 5. Limpieza

La función `limpiar()` (paso 5) aplica, en este orden: minúsculas, quitar HTML, reemplazar enlaces, quitar las palabras de Enron, dejar solo letras y juntar espacios. El orden importa: los enlaces se reemplazan antes de quitar los símbolos, porque después ya estarían partidos en pedazos.

### Una sola función de limpieza para todo

**Qué hicimos:** La misma `limpiar()` se aplica a los correos de entrenamiento, a los correos nuevos del paso 14 y en el modo defensa.

**Por qué:** El modelo tiene que recibir siempre el mismo formato que vio al entrenar. Si un correo nuevo se limpiara distinto, sus palabras no coincidirían con el vocabulario.

**Alternativas que no usamos:** Limpiar de forma distinta en cada etapa.

**Si preguntan:** "Es la misma función en el entrenamiento y en el uso; si no, el modelo recibiría un formato que nunca vio."

**Evidencia:** [Criterio técnico]

### Pasar a minúsculas

**Qué hicimos:** `str(texto).lower()`.

**Por qué:** "URGENT" y "urgent" pasan a ser la misma palabra. El vocabulario se achica y cada palabra junta más ejemplos para aprender.

**Alternativas que no usamos:** Conservar las mayúsculas: escribir "URGENT" en mayúsculas podría ser una señal de presión, y se pierde (inferencia: no lo medimos).

**Si preguntan:** "Para que la misma palabra en mayúscula y en minúscula cuente como una sola."

**Evidencia:** [Criterio técnico]

### Quitar etiquetas HTML

**Qué hicimos:** `re.sub(r'<[^>]+>', ' ', texto)`: todo lo que está entre `<` y `>` se reemplaza por un espacio.

**Por qué:** Las etiquetas son formato, no contenido. Si quedaran, después del filtro de solo letras dejarían palabras falsas, como "div" o "br". Un efecto que hay que conocer: esta regla también borra lo que viene entre `< >` en una línea de remitente. En el correo de soporte TI, `limpiar()` convierte `From: Academic IT Support <support@campus-account-services.invalid>` en `from academic it support`: la dirección desaparece y el modelo no la ve.

**Alternativas que no usamos:** Un analizador de HTML de otra biblioteca: más código para un beneficio pequeño en este proyecto (inferencia).

**Si preguntan:** "Las etiquetas HTML son formato; si las dejo, se convierten en palabras que no significan nada."

**Evidencia:** [Criterio técnico]

### Reemplazar los enlaces por `enlaceweb`

**Qué hicimos:** `re.sub(r'(https?://|www\.)\S+', ' enlaceweb ', texto)`: todo lo que empieza con `http://`, `https://` o `www.`, hasta el siguiente espacio, se convierte en la palabra `enlaceweb`.

**Por qué:** Importa que haya un enlace, no cuál es. Cada dirección es casi única: llenaría el vocabulario de palabras raras, y el filtro de solo letras la partiría en pedazos sin sentido. `enlaceweb` es una palabra inventada, que no existe en inglés, así que no se confunde con una palabra real (criterio técnico). El costo: el modelo no ve el dominio del enlace, que es parte de la explicación probable de por qué se le escapó el correo de soporte TI (interpretación, no medida).

**Alternativas que no usamos:** Conservar el dominio de cada enlace como señal: es más información, pero exige otro tratamiento. Queda como mejora: "sumar el dominio real de los enlaces".

**Si preguntan:** "Lo que delata un fraude es que pida hacer clic, no la dirección exacta. El costo es que no ve dominios sospechosos, y por eso lo pusimos como mejora."

**Evidencia:** [Criterio técnico]

### Quitar las palabras que delatan a Enron

**Qué hicimos:** `re.sub(r'\b(enron|vince|kaminski|hou|ect)\b', ' ', texto)`.

**Por qué:** Son atajos del dataset: el nombre de la empresa (`enron`), nombres internos (`vince`, `kaminski`) y fragmentos de sus direcciones (`hou`, `ect`). Delatan el origen de muchos correos seguros y no generalizarían a otra organización. `\b` significa borde de palabra: así no se borra "ect" dentro de "project". Con honestidad: en el notebook solo se midió `enron` (paso 4); las otras cuatro se quitaron por la misma razón, sin una medición propia en el notebook.

**Alternativas que no usamos:** Dejarlas: el modelo aprendería el atajo. Eliminar los correos que las contienen: perderíamos al menos uno de cada cinco correos seguros (el 20,3% dice `enron`).

**Si preguntan:** "Las quitamos para que el modelo aprenda a reconocer fraude y no a reconocer a Enron."

**Evidencia:** [Medido en el notebook] Paso 4, para `enron`.

### Dejar solo letras y espacios

**Qué hicimos:** `re.sub(r'[^a-záéíóúñü ]', ' ', texto)`, y después `re.sub(r'\s+', ' ', texto).strip()`: los espacios repetidos se juntan en uno y se quitan los de los extremos.

**Por qué:** Números, signos y símbolos agregan muchas variantes con poca información de texto. Con solo letras y un espacio entre palabras, el vectorizador puede separar las palabras simplemente por los espacios. Las letras con tilde, la ñ y la ü se conservan; el notebook no explica por qué, y lo más probable es que sea para no romper palabras en español si se prueba con correos en español (inferencia).

**Alternativas que no usamos:** Conservar números o símbolos, o reemplazarlos por una palabra como hicimos con los enlaces: un monto como el `$2.99` del correo del paquete podría ser una señal de cobro, y se pierde (inferencia: no lo medimos).

**Si preguntan:** "Dejo solo letras porque el modelo lee palabras; el costo es que pierde números y símbolos, como un monto a pagar."

**Evidencia:** [Criterio técnico]

### Eliminar vacíos y duplicados después de limpiar

**Qué hicimos:** `dropna` antes de limpiar. Después, se eliminan los textos que quedaron vacíos y se aplica `drop_duplicates('limpio')`. Quedan 17.103 correos únicos: 10.661 seguros y 6.442 fraudulentos.

**Por qué:** Dos correos pueden volverse iguales solo después de limpiar, por ejemplo si se diferenciaban en un enlace, un número o una mayúscula. Los números lo confirman: de 18.650 filas se descartaron 1.547, más que la suma de vacíos y duplicados exactos del paso 4 (16 + 1.112 = 1.128). Por lo tanto, al menos 419 correos quedaron vacíos o repetidos recién después de limpiar (cálculo con las salidas de los pasos 4 y 5). La separación del paso 6 se hace después de esto, para que un mismo correo no quede a ambos lados.

**Alternativas que no usamos:** Quitar duplicados antes de limpiar: dejaría pasar esos repetidos.

**Si preguntan:** "Quité los duplicados después de limpiar porque la limpieza crea duplicados nuevos, y separé los datos después, para que el mismo correo no quede en entrenamiento y en prueba."

**Evidencia:** [Medido en el notebook] Pasos 4 y 5.

### No quitar palabras vacías ni lematizar

**Qué hicimos:** No se eliminan palabras comunes como "the", "to" o "your" (stopwords), ni se reducen las palabras a su raíz (lematización o stemming).

**Por qué:** La LSTM lee la secuencia. En "click the link below to confirm your password", las palabras pequeñas son parte del orden y del contexto que queremos que pueda considerar. Lematizar exige otra biblioteca y más pasos en vivo. Ninguna de las dos opciones se probó, así que no sabemos si mejorarían el resultado (criterio técnico).

**Alternativas que no usamos:** Quitar stopwords. Lematizar o aplicar stemming.

**Si preguntan:** "No las quitamos porque la LSTM aprovecha el orden y el contexto, y esas palabras son parte de él. No lo probamos, así que no podemos decir que mejoraría."

**Evidencia:** [Criterio técnico]

---

## 6. Separación

### Tres conjuntos: entrenamiento, validación y prueba

**Qué hicimos:** Entrenamiento: 11.629 correos. Validación: 2.053. Prueba: 3.421.

**Por qué:** Con el entrenamiento, la red ajusta sus pesos. Con la validación vigilamos el sobreajuste y EarlyStopping elige la mejor época; no se usa para ajustar pesos. La prueba es la evaluación final, con correos que el modelo nunca vio, y no se toca hasta el final. Como la validación sí influye en una decisión (qué época se queda), hace falta un tercer conjunto que no influya en nada.

**Alternativas que no usamos:** Solo entrenamiento y prueba: habría que elegir la época mirando la prueba, y la prueba dejaría de ser independiente. Validación cruzada: obliga a entrenar varias redes, demasiado lento para una demostración en vivo.

**Si preguntan:** "Entreno con uno, vigilo el sobreajuste con otro y evalúo al final con un tercero que el modelo nunca vio."

**Evidencia:** [Criterio técnico]

### 20% para prueba y 15% del resto para validación

**Qué hicimos:** Primero `test_size=0.2` (3.421 correos de prueba). Después `test_size=0.15` sobre los 13.682 restantes (2.053 de validación y 11.629 de entrenamiento). En total, cerca de 68%, 12% y 20%.

**Por qué:** Son proporciones habituales, no ajustadas. La mayor parte queda para entrenar, y la prueba es grande: con 1.289 fraudes, cada fraude no detectado mueve el recall en menos de una décima de punto porcentual (1 / 1.289, unos 0,08 puntos), así que el resultado no depende de unos pocos correos.

**Alternativas que no usamos:** Otras proporciones, como 70/15/15: no las probamos.

**Si preguntan:** "Son proporciones habituales; no las optimizamos. Lo importante es que la prueba sea grande y no se use hasta el final."

**Evidencia:** [Criterio técnico]

### `stratify`

**Qué hicimos:** `stratify=y` en la primera separación y `stratify=y_resto` en la segunda.

**Por qué:** Mantiene la misma proporción de seguros y fraudes en los tres conjuntos. En la prueba quedan 2.132 seguros y 1.289 fraudes, cerca de 62% y 38%, la misma proporción de los 17.103 correos limpios.

**Alternativas que no usamos:** Separar al azar sin estratificar: la proporción podría cambiar entre conjuntos y las métricas serían menos comparables.

**Si preguntan:** "Stratify mantiene la misma proporción de fraudes en entrenamiento, validación y prueba."

**Evidencia:** [Medido en el notebook] Pasos 5 y 13.

### `random_state=42` en la separación

**Qué hicimos:** La misma semilla, 42, en las dos llamadas a `train_test_split`.

**Por qué:** La separación sale siempre igual. Eso permite repetir el experimento y, sobre todo, reconstruir exactamente la misma prueba en el modo defensa: por eso ahí la exactitud vuelve a dar 0,967. El 42 es una convención; cualquier número fijo sirve.

**Alternativas que no usamos:** Separar sin semilla: cada ejecución tendría otra prueba, y la defensa no podría repetir el resultado.

**Si preguntan:** "Fijo la semilla para que la separación sea siempre la misma; así en la defensa puedo medir el modelo guardado con los mismos correos de prueba."

**Evidencia:** [Medido en el notebook] Modo defensa (exactitud 0,967).

---

## 7. Vectorización

### Convertir palabras en números con `TextVectorization`

**Qué hicimos:** `layers.TextVectorization(max_tokens=20000, output_sequence_length=200, standardize=None)`. Cada palabra se reemplaza por su número en el vocabulario. Ejemplo impreso: "syntax of east asian languages workshop usc workshop on synt" → `[281 3 1056 1661 104 149 3129 149 12 281]`. Los datos de entrenamiento quedan con forma 11.629 × 200.

**Por qué:** Una red no entiende texto, solo números. Al ser una capa de Keras, se puede guardar junto con la red (paso 17). En el ejemplo, "workshop" recibe el mismo número (149) las dos veces: el número identifica la palabra, no su posición. Las palabras más frecuentes reciben números bajos ("of" es el 3), porque Keras ordena el vocabulario por frecuencia.

**Alternativas que no usamos:** Un diccionario hecho a mano en Python: no quedaría guardado con el modelo.

**Si preguntan:** "Cada palabra se cambia por su número en el vocabulario, y cada correo queda como una lista de 200 números."

**Evidencia:** [Medido en el notebook] Paso 7.

### Vocabulario de 20.000

**Qué hicimos:** `max_tokens=20000` (constante `VOCABULARIO`): como máximo 20.000 índices, contando el 0 (relleno) y el 1 (`[UNK]`, palabra fuera del vocabulario).

**Por qué:** Es un valor habitual, no ajustado. Deja fuera las palabras muy raras (nombres, errores de tipeo), que pasan a `[UNK]`, y acota el tamaño del Embedding: 20.000 × 64 = 1.280.000 parámetros, el 97% del modelo. La regresión logística usa el mismo límite (`max_features=VOCABULARIO`), para comparar en igualdad de condiciones.

**Alternativas que no usamos:** El vocabulario completo: muchos más parámetros para palabras que aparecen una o dos veces. Un vocabulario más chico: más palabras convertidas en `[UNK]`. No probamos otros tamaños.

**Si preguntan:** "Es un tamaño habitual: cubre las palabras frecuentes y no infla el Embedding. No lo optimizamos."

**Evidencia:** [Criterio técnico]

### Largo de 200 palabras

**Qué hicimos:** `output_sequence_length=200` (constante `LARGO`). Los correos más cortos se rellenan con ceros al final y los más largos se cortan: solo se leen las primeras 200 palabras.

**Por qué:** La mediana es 160 palabras, así que más de la mitad de los correos cabe completa, y el entrenamiento se mantiene rápido, porque la LSTM da un paso por palabra. Es un equilibrio elegido mirando el paso 4, no un valor optimizado. La limitación: lo que viene después de la palabra 200 no se lee.

**Alternativas que no usamos:** Un largo de 729 o más cubriría al 90% de los correos completos, pero la LSTM daría más del triple de pasos por correo y el entrenamiento sería más lento. Leer el final del correo en vez del inicio: no lo probamos.

**Si preguntan:** "200 cubre más que la mediana y mantiene el entrenamiento rápido. Lo que viene después de la palabra 200 no se lee, y es una limitación."

**Evidencia:** [Medido en el notebook] Paso 4 (mediana de 160 palabras).

### `adapt()` solo con entrenamiento

**Qué hicimos:** `vectorizador.adapt(X_ent)`: el vocabulario se aprende solo con los correos de entrenamiento, y después se aplica a los tres conjuntos.

**Por qué:** Si el vocabulario incluyera validación o prueba, le estaríamos filtrando al modelo información de los correos con que después lo evaluamos. Las palabras de la prueba que no están en el vocabulario pasan a `[UNK]`, igual que le pasaría a un correo real nuevo. TF-IDF sigue la misma regla: `fit_transform` solo con entrenamiento.

**Alternativas que no usamos:** `adapt` con todos los datos: sería una fuga de información.

**Si preguntan:** "El vocabulario sale solo de entrenamiento; si usara la prueba, le estaría filtrando información al modelo."

**Evidencia:** [Criterio técnico]

### `standardize=None`

**Qué hicimos:** Desactivamos la estandarización propia de `TextVectorization`.

**Por qué:** El texto ya viene limpio desde `limpiar()`. Así el vectorizador no cambia nada: lo que produce `limpiar()` es exactamente lo que se convierte en números, tanto en entrenamiento como con correos nuevos. Por defecto, Keras pasaría a minúsculas y quitaría la puntuación, que sería trabajo repetido, y no haría el resto de nuestra limpieza (HTML, enlaces, palabras de Enron).

**Alternativas que no usamos:** Dejar la estandarización por defecto y no usar `limpiar()`: no quitaría HTML, enlaces ni las palabras de Enron.

**Si preguntan:** "Lo desactivé porque ya limpié el texto con mi función; así hay una sola limpieza y siempre es la misma."

**Evidencia:** [Criterio técnico]

### Relleno con 0 y palabra desconocida `[UNK]`

**Qué hicimos:** Usamos los valores por defecto de Keras: el índice 0 es relleno y el índice 1 es `[UNK]`.

**Por qué:** El relleno hace que todos los correos tengan el mismo largo (200), lo que permite procesarlos en lotes. Ese 0 después se ignora gracias a `mask_zero=True` (paso 9). `[UNK]` agrupa las palabras que no están en el vocabulario, en vez de descartarlas sin dejar rastro.

**Alternativas que no usamos:** Correos de largo variable sin relleno: complica armar lotes.

**Si preguntan:** "El 0 es relleno para que todos midan 200, y el 1 es cualquier palabra que no está en el vocabulario."

**Evidencia:** [Criterio técnico]

---

## 8. Línea base

### Medir primero un modelo simple

**Qué hicimos:** Antes de la red entrenamos `TfidfVectorizer(max_features=20000)` + `LogisticRegression(max_iter=1000)`, solo con los 11.629 correos de entrenamiento, y la evaluamos en la misma prueba que la LSTM. Resultado: exactitud 0,967; en fraude, precisión 0,972, recall 0,940 y F1 0,956; AUC 0,995 (el AUC se imprime en el paso 13).

**Por qué:** Un modelo complejo solo se justifica si aporta frente a uno simple, así que primero se mide el simple. Es un modelo visto en el curso que solo mira qué palabras aparecen, no su orden: sirve para saber si la LSTM realmente aporta, y permite responder con datos la pregunta "¿por qué no un modelo más simple?". Se mide antes de la red para fijar la vara de comparación sin conocer el resultado de la LSTM (criterio técnico).

**Alternativas que no usamos:** Otros modelos clásicos, como Naive Bayes o SVM: con una línea base basta, y la regresión logística es un modelo visto en el curso. No comparar con nada: la elección de la red quedaría sin respaldo.

**Si preguntan:** "Primero medimos un modelo simple para saber si la red aportaba. Resultado: rinden prácticamente igual, y lo decimos."

**Evidencia:** [Medido en el notebook] Pasos 8 y 13.

### Qué hacen TF-IDF y la regresión logística

**Qué hicimos:** TF-IDF convierte cada correo en un vector con un peso por palabra (hasta 20.000), y la regresión logística combina esos pesos para estimar la probabilidad de fraude.

**Por qué:** TF-IDF da más peso a las palabras frecuentes en ese correo pero raras en el resto, es decir, a las distintivas. La regresión logística equivale a una sola neurona con sigmoide sobre ese vector, entrenada con la misma pérdida (entropía cruzada) más una regularización que trae por defecto: es el mismo tipo de salida que la última capa de la red, pero sin Embedding ni LSTM, o sea, sin orden. A diferencia de la LSTM, ve el correo completo, sin el corte de 200 palabras. `max_iter=1000` sube el límite de iteraciones del optimizador (por defecto son 100) para que alcance a converger con tantas columnas (criterio técnico).

**Alternativas que no usamos:** Contar palabras sin ponderarlas: TF-IDF es la versión habitual y más informativa.

**Si preguntan:** "La regresión logística es como una neurona con sigmoide sobre un vector de palabras: sabe qué palabras hay, pero no en qué orden."

**Evidencia:** [Criterio técnico]

---

## 9. Arquitectura

Código del paso 9: `keras.Sequential([keras.Input(shape=(LARGO,)), layers.Embedding(VOCABULARIO, 64, mask_zero=True), layers.LSTM(64), layers.Dropout(0.5), layers.Dense(1, activation='sigmoid')])`.

| Capa | Salida | Parámetros |
|---|---|---|
| Embedding | 200 × 64 | 1.280.000 |
| LSTM | 64 | 33.024 |
| Dropout | 64 | 0 |
| Dense | 1 | 65 |
| **Total** | | **1.313.089** (todos entrenables; el 97% está en el Embedding) |

### Una LSTM como modelo principal

**Qué hicimos:** Una red recurrente LSTM (Long Short-Term Memory) con un Embedding antes, para clasificación binaria de texto. La entrada (`Input(shape=(200,))`) son los 200 números de cada correo.

**Por qué:** Un correo es una secuencia de palabras, y el orden puede importar: "click the link below to confirm your password" no es lo mismo que esas palabras desordenadas. Una LSTM lee palabra por palabra y guarda una memoria de lo leído. Con sus compuertas puede conservar información durante muchos pasos, algo que una RNN simple pierde, porque en ella el gradiente se desvanece en secuencias largas y olvida las palabras lejanas. Se basa en el ejemplo de noticias falsas del profesor, que usa los mismos componentes (Embedding de 64, LSTM, Dropout de 0,5 y Dense con sigmoide) pero con dos LSTM apiladas; nosotros usamos una (ver la sección 25). Ojo: la elegimos porque puede considerar el orden, no porque haya rendido más (ver la sección 21).

**Alternativas que no usamos:**
- RNN simple: olvida las palabras lejanas, por el gradiente que se desvanece.
- GRU: parecida a la LSTM, con menos parámetros. Nos quedamos con la LSTM del ejemplo del curso; la GRU no la probamos.
- LSTM bidireccional o varias LSTM apiladas (el ejemplo de noticias falsas apila dos): más parámetros y más tiempo de entrenamiento en una demostración en vivo de 15 minutos. Con el empate medido, no hay evidencia de que haga falta más capacidad (inferencia).
- Red convolucional 1D sobre texto: también es una opción para secuencias; no la probamos.
- Transformers o BERT preentrenado: mucho más grandes, más pesados de descargar y de ajustar en vivo, y su teoría no cabe en la explicación de 15 minutos.

**Si preguntan:** "Elegimos LSTM porque el correo es una secuencia y la LSTM puede considerar el orden de las palabras. No decimos que gane: en este dataset empata con la regresión logística."

**Evidencia:** [Criterio técnico]

### Embedding de 64 dimensiones

**Qué hicimos:** `Embedding(20000, 64)`: una tabla entrenable de 20.000 × 64 números (1.280.000 parámetros). Cada índice de palabra se convierte en un vector de 64 valores. La tabla se aprende junto con el resto de la red, desde cero, y tiene una fila por cada índice que puede entregar el vectorizador (por eso usa el mismo `VOCABULARIO`).

**Por qué:** Un número de palabra, como 149, no dice nada por sí solo; un vector sí puede: las palabras que se usan en contextos parecidos terminan con vectores parecidos. 64 es un tamaño habitual en modelos chicos, no ajustado: alcanza para representar palabras sin inflar los parámetros (con 128, el Embedding se duplicaría a 2.560.000).

**Alternativas que no usamos:** Embeddings preentrenados, como word2vec o GloVe: hay que descargarlos aparte y adaptarlos al vocabulario; no los probamos. One-hot: vectores de 20.000 posiciones casi todas en cero, muy grandes y sin noción de parecido entre palabras.

**Si preguntan:** "El Embedding es una tabla que la red aprende: a cada palabra le asigna 64 números, y las palabras de uso parecido quedan con números parecidos."

**Evidencia:** [Criterio técnico]

### `mask_zero=True`

**Qué hicimos:** El Embedding marca el índice 0 como relleno, y la LSTM salta esas posiciones.

**Por qué:** El relleno va al final del correo. Sin máscara, la LSTM seguiría leyendo relleno después de la última palabra real, hasta completar los 200 pasos, y eso podría diluir lo que recordaba (criterio técnico). En el ensayo se comprobó que, con la máscara, la salida de la LSTM es igual a la de leer solo las palabras reales, sin relleno. Si Keras muestra un aviso sobre `mask`, se refiere a que las capas siguientes no reciben la máscara; la LSTM la usa igual.

**Alternativas que no usamos:** Entrenar sin máscara, dejando que la red aprenda sola a ignorar los ceros.

**Si preguntan:** "Los ceros son relleno, no palabras, y con la máscara la LSTM los ignora. Lo comprobamos: da lo mismo que leer solo las palabras reales."

**Evidencia:** [Comprobado en ensayo] Script de verificación `verif.py`.

### LSTM de 64 unidades

**Qué hicimos:** `LSTM(64)`: lee los vectores en orden y entrega solo el último estado oculto, 64 valores que resumen el correo. Tiene 33.024 parámetros: 4 bloques de pesos × (64 × (64 + 64) + 64).

**Por qué:** En cada palabra, la LSTM mantiene un estado de celda (la memoria) y un estado oculto. La compuerta de olvido (sigmoide) decide qué borrar de la memoria; la compuerta de entrada (sigmoide) y un candidato (tanh) deciden qué agregar; la compuerta de salida (sigmoide) decide qué parte de la memoria, pasada por tanh, se entrega: `c_t = f_t * c_(t-1) + i_t * c~_t` y `h_t = o_t * tanh(c_t)`. Como la memoria se actualiza sumando, la información y el gradiente pueden recorrer muchos pasos. Cada compuerta y el candidato tienen su propio bloque de pesos, y de ahí sale el 4 de la cuenta. Solo usamos el último estado porque la tarea es una sola decisión por correo. 64 unidades es un valor habitual, no ajustado, igual al tamaño del Embedding.

**Alternativas que no usamos:** Entregar el estado de cada palabra (`return_sequences=True`) para otra capa encima: no hace falta para una sola decisión. Más unidades: más parámetros, sin evidencia de que hagan falta.

**Si preguntan:** "La LSTM tiene una memoria y tres compuertas: qué olvidar, qué agregar y qué entregar. Al final entrega 64 números que resumen el correo."

**Evidencia:** [Medido en el notebook] Paso 9 (33.024 parámetros en el `summary`).

### Dropout de 0,5

**Qué hicimos:** `Dropout(0.5)` entre la LSTM y la neurona de salida.

**Por qué:** Solo durante el entrenamiento, en cada paso pone en cero al azar la mitad de los 64 valores que salen de la LSTM, y multiplica el resto por 2 para compensar. Así la red no puede depender de unas pocas unidades, y eso reduce el sobreajuste. Al predecir no se apaga nada. 0,5 es un valor habitual, no ajustado. Ojo con la palabra: apaga valores (las salidas de las unidades), no conexiones; cortar conexiones es otra técnica, llamada DropConnect.

**Alternativas que no usamos:** Otro porcentaje, como 0,2 o 0,3. Dropout dentro de la LSTM (sus opciones `dropout` y `recurrent_dropout`). Regularización L2 de los pesos. No probamos ninguna.

**Si preguntan:** "Al entrenar apaga al azar la mitad de los valores que salen de la LSTM, para que la red no dependa de unos pocos; al predecir no apaga nada."

**Evidencia:** [Criterio técnico]

### Salida `Dense(1, activation='sigmoid')`

**Qué hicimos:** Una neurona con activación sigmoide: `σ(z) = 1 / (1 + e^(-z))`. Tiene 65 parámetros: 64 pesos y 1 sesgo.

**Por qué:** El problema es binario, así que basta un número entre 0 y 1, que se lee como la probabilidad de fraude. Esta neurona es el ejemplo directo de neurona artificial: calcula `σ(w1·x1 + ... + w64·x64 + b)` sobre los 64 valores de la LSTM. Los pesos dicen cuánto importa cada valor, el sesgo desplaza el resultado y la sigmoide lo transforma en probabilidad. Con umbral 0,5, la frontera queda en z = 0, porque σ(0) = 0,5.

**Alternativas que no usamos:** Dos neuronas con softmax y entropía cruzada categórica: para dos clases es equivalente y más difícil de explicar.

**Si preguntan:** "Una sola neurona con sigmoide, porque necesito una probabilidad entre 0 y 1 para dos clases."

**Evidencia:** [Criterio técnico]

---

## 10. Compilación

### Optimizador Adam

**Qué hicimos:** `optimizer='adam'`, con la tasa de aprendizaje por defecto de Keras (0,001).

**Por qué:** Adam es una forma de descenso del gradiente. La retropropagación calcula el gradiente de la pérdida respecto de cada peso, desde la salida hacia la entrada; en la LSTM también recorre hacia atrás las palabras de la secuencia (retropropagación a través del tiempo). Después, cada peso se mueve un pequeño paso en contra de su gradiente. Adam adapta el tamaño de ese paso para cada peso, usando promedios móviles del gradiente y de su cuadrado. Suele funcionar bien sin ajustar la tasa de aprendizaje, lo que conviene en una demostración en vivo (criterio técnico).

**Alternativas que no usamos:** Descenso del gradiente clásico (SGD): habría que elegir y probar una tasa de aprendizaje. RMSprop. No probamos ninguno.

**Si preguntan:** "Adam es descenso del gradiente con un paso que se adapta a cada peso; usé su tasa por defecto, 0,001."

**Evidencia:** [Criterio técnico]

### Pérdida: entropía cruzada binaria

**Qué hicimos:** `loss='binary_crossentropy'`: `L = -[y·log(p) + (1-y)·log(1-p)]`, con logaritmo natural, promediada en cada lote.

**Por qué:** Es la pérdida que corresponde a una salida sigmoide en un problema de dos clases. Mide el error (qué tan lejos está la probabilidad de la respuesta real) y castiga fuerte equivocarse con seguridad: un fraude (y = 1) con p = 0,9 cuesta 0,11, y con p = 0,1 cuesta 2,30. Así la red no solo busca acertar, también evita equivocarse con confianza.

**Alternativas que no usamos:** Error cuadrático medio: con sigmoide da gradientes débiles justo cuando la red se equivoca mucho, así que aprende más lento (criterio técnico). Pérdida focal o pesos por clase: están pensados para desbalances fuertes, y el nuestro es moderado.

**Si preguntan:** "Mide qué tan lejos está la probabilidad de la respuesta real y castiga mucho los errores confiados: decirle 0,1 a un fraude cuesta 2,30; decirle 0,9 cuesta 0,11."

**Evidencia:** [Criterio técnico]

### Métricas: exactitud, recall y precisión

**Qué hicimos:** `metrics=['accuracy', keras.metrics.Recall(name='recall'), keras.metrics.Precision(name='precision')]`.

**Por qué:** El recall es VP / (VP + FN): de todos los fraudes, cuántos detecta. Importa porque un fraude no detectado puede terminar en robo de claves, que es el error caro. La precisión es VP / (VP + FP): de lo que marca como fraude, cuánto lo es. Importa porque una falsa alarma puede esconder un correo legítimo y le quita confianza al filtro. La exactitud sola engaña con clases desbalanceadas. El parámetro `name=` deja nombres cortos (`recall`, `val_recall`) en la salida del entrenamiento.

**Alternativas que no usamos:** Medir solo exactitud.

**Si preguntan:** "Mido recall porque un fraude que pasa es el error caro, y precisión porque una falsa alarma puede esconder un correo legítimo."

**Evidencia:** [Criterio técnico]

---

## 11. Entrenamiento

Código del paso 11: `modelo.fit(X_ent_num, y_ent, validation_data=(X_val_num, y_val), epochs=6, batch_size=32, callbacks=[parada], verbose=2)`.

### Lotes (batch) de 32 correos

**Qué hicimos:** `batch_size=32`: los pesos se actualizan cada 32 correos. Con 11.629 correos de entrenamiento son 364 actualizaciones por época (el último lote tiene 13 correos).

**Por qué:** En cada lote se repite el ciclo: propagación hacia adelante, cálculo de la pérdida, retropropagación y actualización de Adam. 32 es el valor por defecto de Keras y un estándar, no ajustado. Es un punto medio: actualizar con un correo a la vez es muy ruidoso y lento, y con todos a la vez habría muy pocas actualizaciones y haría falta mucha memoria.

**Alternativas que no usamos:** Lotes de 64 o 128: menos actualizaciones por época y más rápido en GPU. No los probamos.

**Si preguntan:** "Cada 32 correos la red calcula la pérdida, retropropaga el error y Adam actualiza los pesos: 364 veces por época."

**Evidencia:** [Medido en el notebook] Paso 11 ("364/364" en cada época).

### Máximo de 6 épocas

**Qué hicimos:** `epochs=6`.

**Por qué:** Una época es una pasada completa por los correos de entrenamiento. 6 es un tope elegido para que el entrenamiento quepa en el video, no un valor ajustado: en CPU, cada época tomó unos 17 a 18 segundos. Además, el tope casi no importa, porque EarlyStopping decide cuándo parar: en el ensayo paró después de la cuarta.

**Alternativas que no usamos:** Más épocas: más tiempo en vivo y más riesgo de sobreajuste, aunque EarlyStopping lo controlaría.

**Si preguntan:** "Una época es una pasada por todos los correos de entrenamiento. Puse un máximo de 6 por tiempo, y la parada temprana decide antes si conviene."

**Evidencia:** [Medido en el notebook] Paso 11.

### EarlyStopping con `val_loss`, paciencia 2 y mejores pesos

**Qué hicimos:** `keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)`.

**Por qué:** El sobreajuste es cuando la red memoriza los correos de entrenamiento y empeora con correos nuevos; su señal es que la pérdida de entrenamiento sigue bajando mientras la de validación sube. EarlyStopping corta ahí: si la pérdida de validación no mejora en 2 épocas seguidas, detiene el entrenamiento y recupera los pesos de la mejor época. Vigila la pérdida y no la exactitud porque la pérdida es continua y refleja la confianza, mientras la exactitud cambia a saltos (criterio técnico). La paciencia de 2 tolera una época mala, porque la validación salta entre épocas, sin gastar tiempo de más. Sin `restore_best_weights`, el modelo quedaría con los pesos de la última época, que eran peores. EarlyStopping no aparece en el material del curso: se explica con los conceptos de época y sobreajuste, que sí pide el enunciado.

**Alternativas que no usamos:** Un número fijo de épocas, sin parada. Vigilar `val_accuracy`. Una paciencia mayor (más épocas en vivo).

**Si preguntan:** "Si el error en validación no mejora en dos épocas, paro y vuelvo a los mejores pesos. Junto con Dropout y la validación separada, es como controlamos el sobreajuste."

**Evidencia:** [Medido en el notebook] Paso 11.

### Validación explícita y una línea por época

**Qué hicimos:** `validation_data=(X_val_num, y_val)` y `verbose=2`.

**Por qué:** Le pasamos a Keras nuestro conjunto de validación, ya estratificado en el paso 6. `verbose=2` imprime una sola línea por época, sin barra de progreso, que se lee bien en el video.

**Alternativas que no usamos:** `validation_split`, que separa la validación automáticamente desde los datos de entrenamiento sin estratificar.

**Si preguntan:** "Uso mi propia validación, separada y estratificada, para que EarlyStopping mire correos que no se usan para ajustar pesos."

**Evidencia:** [Criterio técnico]

### Qué pasó en el ensayo

**Qué hicimos:** La pérdida de validación por época fue 0,532 → 0,111 → 0,168 → 0,122. La mejor fue la época 2 (val_accuracy 0,969). Como las épocas 3 y 4 no mejoraron, el entrenamiento se detuvo después de la 4 y recuperó los pesos de la 2. En CPU tomó unos 17 a 18 segundos por época, poco más de 1 minuto en total.

**Por qué:** Muestra que EarlyStopping hizo su trabajo. En la época 4, la pérdida de entrenamiento bajó a 0,070 mientras la de validación quedó en 0,122, por encima de su mejor valor: es la señal típica de que empieza el sobreajuste, y justo lo que EarlyStopping evita (interpretación nuestra). La época 3 muestra además que el entrenamiento no es perfectamente estable: la pérdida de entrenamiento subió de 0,148 a 0,198.

**Alternativas que no usamos:** No aplica: es un resultado, no una elección.

**Si preguntan:** "Paró en la época 4 y recuperó la 2, que tenía el menor error de validación. En la grabación puede parar en otra época, y es normal."

**Evidencia:** [Medido en el notebook] Paso 11.

---

## 12. Curvas de aprendizaje

### Graficar pérdida y exactitud por época

**Qué hicimos:** Dos gráficos lado a lado (`figsize=(11, 3.5)`): pérdida y exactitud por época, entrenamiento contra validación, a partir de `historial.history`. Es código auxiliar: se pega, pero se explica igual.

**Por qué:** Es la forma visual de ver el sobreajuste: si la validación empeora mientras el entrenamiento mejora, la red está memorizando. Hay dos detalles para no confundirse. Primero, el eje x parte en 0, así que el punto 0 es la época 1. Segundo, las curvas muestran todas las épocas que corrieron, incluidas las que vinieron después de la mejor, aunque el modelo final usa los pesos de la mejor época.

**Alternativas que no usamos:** Mirar solo los números del paso 11: el gráfico muestra la separación entre curvas de un vistazo.

**Si preguntan:** "Si la curva de validación se separa hacia arriba mientras la de entrenamiento baja, hay sobreajuste. Por eso existe la parada temprana." Y si preguntan por qué en la época 2 la validación salió mejor que el entrenamiento (0,111 contra 0,148): "La pérdida de entrenamiento es el promedio durante la época, con Dropout activo; la de validación se mide al final de la época y sin Dropout." (criterio técnico)

**Evidencia:** [Criterio técnico]

---

## 13. Evaluación

### Evaluar una sola vez con la prueba, con umbral 0,5

**Qué hicimos:** `modelo.predict(X_prueba_num)` y umbral 0,5 sobre los 3.421 correos de prueba, que no se usaron para nada antes. Resultado de la LSTM: exactitud 0,967; en seguro, precisión 0,967 y recall 0,981; en fraude, precisión 0,967, recall 0,945 y F1 0,956.

**Por qué:** El reporte por clase (`classification_report`, con 3 decimales) muestra el rendimiento en cada clase y no solo un promedio. El F1 es la media armónica de precisión y recall: solo es alto si los dos lo son.

**Alternativas que no usamos:** Evaluar con la validación: ya se usó para elegir la época, así que no es independiente.

**Si preguntan:** "La prueba se usa una sola vez, al final. El F1 combina precisión y recall, y solo sube si los dos son buenos."

**Evidencia:** [Medido en el notebook] Paso 13.

### AUC

**Qué hicimos:** `roc_auc_score` para la LSTM (0,991) y para la regresión logística (0,995).

**Por qué:** El AUC es la probabilidad de que un fraude elegido al azar reciba un puntaje más alto que un correo seguro elegido al azar (1 es perfecto; 0,5 es azar). No depende del umbral: mide qué tan bien ordena el modelo, no solo cómo corta en 0,5. Por eso es una comparación justa entre los dos modelos.

**Alternativas que no usamos:** Comparar solo con el umbral 0,5: dependería de ese corte.

**Si preguntan:** "El AUC dice qué tan bien separa fraudes de seguros con cualquier umbral; la regresión logística sacó 0,995 y la LSTM 0,991."

**Evidencia:** [Medido en el notebook] Paso 13.

### Matriz de confusión

**Qué hicimos:** `ConfusionMatrixDisplay` (con la paleta de colores `Blues`). Las filas son la clase real y las columnas la predicción. De 2.132 seguros, 2.091 quedaron bien y 41 fueron marcados por error (arriba a la derecha). De 1.289 fraudes, 1.218 fueron detectados y 71 no (abajo a la izquierda).

**Por qué:** Pasa de porcentajes a correos concretos, que es el lenguaje del problema. Además distingue los dos errores: el falso negativo (un fraude que pasa como seguro) es el caro; el falso positivo (un correo seguro marcado) le quita confianza al filtro y puede esconder un correo legítimo.

**Alternativas que no usamos:** Reportar solo porcentajes.

**Si preguntan:** "La diagonal son aciertos. Abajo a la izquierda están los 71 fraudes que se escaparon, que es el error más caro."

**Evidencia:** [Medido en el notebook] Paso 13.

### Por qué la exactitud sola engaña

**Qué hicimos:** Reportamos recall, precisión, F1, AUC y la matriz, además de la exactitud.

**Por qué:** Responder siempre "seguro" ya acertaría cerca del 62% en la prueba (2.132 de 3.421) sin detectar ni un fraude. Una exactitud alta no dice qué errores comete el modelo.

**Alternativas que no usamos:** Reportar solo la exactitud.

**Si preguntan:** "Un modelo que dijera siempre 'seguro' tendría 62% de exactitud y no serviría para nada; por eso miro recall y precisión."

**Evidencia:** [Medido en el notebook] Paso 13.

### Comparar con la línea base en la misma prueba

**Qué hicimos:** En el paso 13 se imprimen juntas la exactitud (0,967 en los dos modelos) y el AUC (0,991 la LSTM y 0,995 la regresión logística).

**Por qué:** Es la comparación honesta: en este dataset, la LSTM rinde prácticamente igual que la regresión logística. Lo decimos sin rodeos (ver la sección 21).

**Alternativas que no usamos:** Mostrar solo la LSTM.

**Si preguntan:** "En este dataset empatan, y lo mostramos; esa comparación es la que justifica o no un modelo más complejo."

**Evidencia:** [Medido en el notebook] Pasos 8 y 13.

---

## 14. Prueba con correos nuevos

### La función `clasificar_correo()`

**Qué hicimos:** `clasificar_correo(texto, umbral=0.5)`: limpia el texto con `limpiar()`, lo convierte con el mismo `vectorizador`, predice con `modelo.predict` y devuelve la probabilidad y la etiqueta (`FRAUDULENTO` o `SEGURO`).

**Por qué:** El enunciado pide al menos una prueba con datos que demuestre el funcionamiento. Usar la misma limpieza y el mismo vectorizador del entrenamiento asegura que el correo nuevo llegue en el mismo formato. Dejar el umbral como parámetro permite cambiarlo sin tocar la función.

**Alternativas que no usamos:** Probar solo con la prueba del dataset: no mostraría el modelo frente a un correo escrito por nosotros.

**Si preguntan:** "La función hace con el correo nuevo exactamente lo mismo que se hizo con los de entrenamiento, y entrega la probabilidad de fraude."

**Evidencia:** [Requisito del enunciado]

### Por qué esos 4 correos

**Qué hicimos:** Cuatro correos escritos para esta prueba, que no vienen del dataset: Banco (fraude con urgencia y amenaza de suspensión), Paquete (fraude con un cobro pequeño y un plazo), Soporte TI (fraude bien escrito, sin urgencia ni amenazas) y Coordinación (aviso académico legítimo). Resultado: Banco, FRAUDULENTO 99,1%; Paquete, FRAUDULENTO 97,8%; Soporte TI, SEGURO 4,1% (se le escapó); Coordinación, SEGURO 0,4%.

**Por qué:** Cubren los casos que importan: dos fraudes evidentes, un fraude difícil que pone a prueba el límite del modelo y un correo legítimo que muestra que no marca todo como fraude. Soporte TI y Coordinación usan el contexto universitario, que también es parte del problema.

**Alternativas que no usamos:** Correos del dataset: esos ya los mide la prueba. Correos reales: tendrían datos personales y enlaces reales.

**Si preguntan:** "El 3 lo pusimos a propósito: es el fraude difícil. Se le escapó y lo decimos, porque muestra su límite real."

**Evidencia:** [Medido en el notebook] Paso 14.

### Correos en inglés

**Qué hicimos:** Los 4 correos están en inglés.

**Por qué:** El dataset y el vocabulario están en inglés. En un correo en español, la mayoría de las palabras no estaría en el vocabulario y pasaría a `[UNK]`, así que el resultado diría poco del modelo (inferencia: no lo probamos en el notebook).

**Alternativas que no usamos:** Correos en español: tienen sentido cuando se entrene con datos en español, que es una de las mejoras.

**Si preguntan:** "Están en inglés porque el modelo aprendió de correos en inglés; probarlo en español sería probar algo que no aprendió."

**Evidencia:** [Criterio técnico]

### Enlaces y remitentes con el dominio `.invalid`

**Qué hicimos:** Los enlaces y las direcciones de remitente usan el dominio reservado `.invalid`, que no existe.

**Por qué:** En el video no aparece ningún enlace a un sitio real, ni de un banco ni de un atacante: nadie puede hacer clic y llegar a algo real. No cambia lo que ve el modelo: los enlaces se convierten en `enlaceweb`, y las direcciones de remitente, que van entre `< >`, las borra la regla de HTML de `limpiar()`.

**Alternativas que no usamos:** Dominios inventados que podrían existir o pertenecer a alguien.

**Si preguntan:** "Usamos un dominio que no existe para no mostrar enlaces reales; al modelo le da lo mismo, porque no ve el dominio."

**Evidencia:** [Criterio técnico]

### Mostrar la probabilidad, no solo la etiqueta

**Qué hicimos:** Se imprime `P(fraude)` con un decimal junto a la decisión.

**Por qué:** La probabilidad dice qué tan seguro está el modelo, y le permite a la organización elegir el umbral y priorizar qué revisar. En el correo de soporte TI muestra algo importante: no fue un caso dudoso, el modelo estaba muy seguro y se equivocó (4,1%). Bajar el umbral a 0,3 no lo habría atrapado; habría que bajarlo de 0,041 (cálculo directo).

**Alternativas que no usamos:** Mostrar solo FRAUDULENTO o SEGURO: se perdería la información de confianza.

**Si preguntan:** "La probabilidad dice cuánto confía el modelo. En el correo de soporte TI se equivocó con mucha seguridad, así que mover el umbral no lo arregla."

**Evidencia:** [Medido en el notebook] Paso 14.

### Mantener en la prueba el correo que el modelo falla

**Qué hicimos:** El correo de soporte TI se dejó en la prueba aunque el modelo lo clasifica como seguro.

**Por qué:** Muestra el límite real del modelo. Explicación probable (interpretación, no medida): el modelo se apoya en el lenguaje de presión (urgencia, amenazas, cobros), y este correo no lo tiene; además no ve el dominio sospechoso, porque el enlace se convierte en `enlaceweb`, la dirección del remitente se borra al limpiar y el modelo no analiza el remitente. De aquí sale la mejora concreta: analizar el remitente y el dominio real de los enlaces.

**Alternativas que no usamos:** Cambiarlo por un correo que el modelo acierte: la demostración se vería mejor, pero sería engañosa.

**Si preguntan:** "Lo dejamos porque es honesto: es el tipo de fraude que hoy se le escapa, y nos dice qué mejorar."

**Evidencia:** [Medido en el notebook] Paso 14.

---

## 15. Interpretación

### Traducir las métricas al lenguaje del problema

**Qué hicimos:** A partir de la matriz de confusión (`vn, fp, fn, vp = confusion_matrix(...).ravel()`): de 1.289 correos fraudulentos, detectó 1.218 (94,5%) y dejó pasar 71; de 2.132 correos seguros, 41 (1,9%) quedarían marcados por error.

**Por qué:** El enunciado pide interpretar el resultado, relacionarlo con el problema del integrante 1 y responder qué evidencia demuestra que el modelo contribuye a resolverlo. Dicho en simple: de cada 100 fraudes, unos 5 llegarían a la bandeja; de cada 100 correos legítimos, unos 2 irían a cuarentena sin serlo. Por eso es un filtro de apoyo: lo marcado se revisa, no se borra, y los fraudes que se escapan obligan a mantener la capacitación y otros controles.

**Alternativas que no usamos:** Terminar en las métricas, sin traducirlas al problema.

**Si preguntan:** "De cada 100 fraudes detecta unos 95, y de cada 100 correos buenos manda unos 2 a revisión. Ayuda, pero no reemplaza otros controles."

**Evidencia:** [Requisito del enunciado]

---

## 16. Umbral

### Umbral de 0,5 y tabla opcional de umbrales

**Qué hicimos:** La evaluación principal usa 0,5. El paso 16 (opcional, fuera de los 15 minutos) prueba cuatro umbrales: 0,3 detecta 95,7% del fraude con 2,4% de falsas alarmas; 0,5, 94,5% con 1,9%; 0,7, 92,1% con 1,5%; 0,9, 85,9% con 0,8%.

**Por qué:** 0,5 es el punto neutro de la sigmoide: marca fraude cuando es más probable que sí a que no. Pero el umbral es una decisión de la organización, según qué error le cuesta más: bajarlo detecta más fraude a cambio de más falsas alarmas. Con cuarentena y revisión humana, una falsa alarma cuesta tiempo de revisión, así que un umbral más bajo podría convenir; no tenemos los costos reales de una organización para decidirlo (inferencia). Los cuatro valores muestran el rango; no salen de una optimización.

**Alternativas que no usamos:** Elegir el "mejor" umbral mirando la prueba: sería ajustar el modelo con la prueba. Dibujar la curva ROC completa: más tiempo para la misma idea.

**Si preguntan:** "El umbral no lo decide el modelo, lo decide la organización: pasar de 0,5 a 0,3 detecta 1,2 puntos más de fraude a cambio de 0,5 puntos más de falsas alarmas."

**Evidencia:** [Medido en el notebook] Paso 16.

---

## 17. Guardar el modelo completo y modo defensa

### Guardar el vectorizador y la red en un solo archivo

**Qué hicimos:** `modelo_completo = keras.Sequential([keras.Input(shape=(1,), dtype='string'), vectorizador, modelo])` y `modelo_completo.save('modelo_phishing.keras')`. En Colab, `files.download` descarga el archivo al computador; fuera de Colab, el `try/except ImportError` evita que esa línea falle.

**Por qué:** El enunciado pide presentar nuevamente en la defensa el modelo completamente operativo: que reciba los datos, los procese y genere resultados coherentes. Juntar el vectorizador con la red hace que el archivo reciba texto directamente y traiga su propio vocabulario, así que no hay que volver a entrenar ni recrear el vocabulario.

**Alternativas que no usamos:** Guardar la red sola y el vocabulario aparte: son dos archivos que deben coincidir. Volver a entrenar en la defensa: toma tiempo y los resultados pueden variar.

**Si preguntan:** "Guardamos el vectorizador junto con la red; así el archivo recibe texto y trae su vocabulario."

**Evidencia:** [Requisito del enunciado]

### Modo defensa: cargar sin reentrenar

**Qué hicimos:** Se ejecutan los pasos 1, 2, 3, 5 y 6, se sube `modelo_phishing.keras` y se ejecuta la celda "Modo defensa". Esa celda carga el modelo, vuelve a medir la exactitud en la prueba (0,967) y clasifica la frase "Your account has been suspended. Click here and confirm your password now." con 0,987 de probabilidad: FRAUDULENTO.

**Por qué:** El archivo trae el vocabulario y la red, pero no la función `limpiar()`, que es código Python: por eso hay que ejecutar el paso 5 antes. Los pasos 3 y 6 rehacen la misma prueba, gracias a `random_state=42`, para mostrar que el modelo cargado da el mismo resultado. El paso 7 no hace falta, porque el vectorizador viene dentro del archivo. La entrada va con forma `(n, 1)` (`tf.constant([[...]])` y `X_prueba.reshape(-1, 1)`) porque el modelo completo espera un texto por fila.

**Alternativas que no usamos:** Mostrar solo resultados guardados: el enunciado pide demostrar el funcionamiento real.

**Si preguntan:** "Cargo el modelo guardado sin reentrenar: da la misma exactitud, 0,967, y clasifica en el momento un texto que escribo."

**Evidencia:** [Medido en el notebook] Celda "Modo defensa".

---

## 18. Semilla y reproducibilidad

### `keras.utils.set_random_seed(42)`

**Qué hicimos:** En el paso 2 fijamos la semilla de Python, NumPy y TensorFlow con `keras.utils.set_random_seed(42)`. Las separaciones usan además `random_state=42`.

**Por qué:** La red tiene partes al azar: los pesos iniciales, el orden de los lotes y Dropout. Fijar la semilla permite repetir el experimento. El 42 es una convención; cualquier número fijo sirve. Aun así, los resultados pueden variar un poco entre ejecuciones y entre CPU y GPU, así que en la grabación pueden salir números levemente distintos a los del notebook de respaldo: se interpretan los que salgan.

**Alternativas que no usamos:** No fijar semilla: resultados distintos en cada ejecución. Forzar un determinismo estricto en TensorFlow: más lento, y no hacía falta para este trabajo (criterio técnico).

**Si preguntan:** "La semilla hace repetible el experimento. Si en vivo cambian un poco los números, se debe a diferencias entre ejecuciones o entre CPU y GPU, no a un error."

**Evidencia:** [Criterio técnico]

---

## 19. Conclusión: "parcialmente"

### Responder que la solución resuelve el problema parcialmente

**Qué hicimos:** La conclusión del notebook dice que la solución resuelve el problema parcialmente.

**Por qué:** El enunciado pide terminar el video diciendo si la solución responde total o parcialmente al problema, y cuáles son sus principales mejoras. Funciona como filtro de apoyo contra el fraude evidente (detectó 94,5% de los fraudes de prueba, con 1,9% de falsas alarmas). Pero dejó pasar 71 fraudes de prueba y el correo de soporte TI, que estaba bien escrito; la regresión logística rinde prácticamente igual; y los datos tienen límites (inglés, mezcla con spam, sesgo hacia Enron). Con esa evidencia, decir "totalmente" no se sostiene.

**Alternativas que no usamos:** Decir "totalmente".

**Si preguntan:** "Lo resuelve parcialmente: sirve como filtro de apoyo contra el fraude evidente, pero se le escapan fraudes bien escritos, así que no reemplaza otros controles."

**Evidencia:** [Requisito del enunciado]

---

## 20. Ética y ley

### Cuidar la privacidad del contenido de los correos

**Qué hicimos:** Entre las mejoras, la conclusión del notebook incluye cuidar la privacidad del contenido de los correos, y menciona la Ley 21.719, de protección de datos personales.

**Por qué:** El modelo lee el texto de los correos, que puede contener datos personales. Usarlo en una organización real exige tratar ese contenido con cuidado y conforme a esa ley. En el proyecto no usamos correos de personas reales: el dataset es público y los 4 correos de prueba se escribieron para la prueba, con dominios que no existen.

**Alternativas que no usamos:** Entrenar o probar con correos reales de la universidad o de una empresa.

**Si preguntan:** "Un filtro así lee contenido privado, así que tiene que cumplir la ley de protección de datos personales. Nosotros trabajamos con datos públicos y correos inventados."

**Evidencia:** [Criterio técnico]

### Revisión humana y transparencia sobre los errores

**Qué hicimos:** Proponemos cuarentena con revisión humana en vez de bloqueo automático, y reportamos los errores del modelo (71 fraudes no detectados y 41 falsas alarmas en la prueba) y el empate con la regresión logística.

**Por qué:** Es parte de la ética profesional que pide el criterio 2.1.2: un sistema que se equivoca no debería decidir solo sobre la comunicación de las personas, y la organización debe conocer sus límites para no confiar de más en él.

**Alternativas que no usamos:** Presentar el modelo como infalible o esconder el empate.

**Si preguntan:** "El modelo apoya, no decide: lo dudoso lo revisa una persona, y decimos con claridad en qué se equivoca."

**Evidencia:** [Requisito del enunciado]

---

## 21. La pregunta difícil: ¿por qué Deep Learning si la regresión logística empata?

Es la pregunta más probable de la defensa, para los dos. La respuesta tiene cuatro partes: lo medido, por qué puede pasar, qué ofrece la LSTM y qué experimento lo comprobaría.

### Lo que medimos

| En la prueba (3.421 correos) | LSTM | Regresión logística |
|---|---|---|
| Exactitud | 0,967 | 0,967 |
| AUC | 0,991 | 0,995 |
| Recall de fraude | 0,945 | 0,940 |
| Precisión de fraude | 0,967 | 0,972 |
| F1 de fraude | 0,956 | 0,956 |
| Fraudes no detectados | 71 | 77 (deducido) |
| Seguros marcados por error | 41 | 35 (deducido) |
| Errores en total | 112 | 112 (deducido) |

El notebook no imprime la matriz de la regresión logística. Los valores marcados como deducidos salen de su reporte en el paso 8: por ejemplo, un recall de 0,940 sobre 1.289 fraudes deja unos 77 sin detectar. Es la única combinación compatible con todas las cifras de ese reporte.

Lectura: los dos modelos se equivocan en la misma cantidad de correos. La LSTM detecta 6 fraudes más, pero marca 6 correos seguros más por error, y la regresión logística tiene un AUC algo mayor. Son diferencias de pocos correos en una sola ejecución: no permiten decir que uno sea mejor. Es un empate.

### Por qué puede pasar en este dataset (inferencias, no medidas)

- **Las palabras solas ya separan casi todo.** El spam y el phishing de este dataset usan un vocabulario muy distinto al de los correos de trabajo, y una regresión logística sobre TF-IDF aprovecha justamente eso.
- **Pocos datos para una red.** La LSTM aprende 1.313.089 parámetros desde cero (casi todos en el Embedding) con unos 11.600 correos de entrenamiento. La regresión logística recibe de TF-IDF las palabras ya ponderadas y aprende muchos menos parámetros.
- **El corte de 200 palabras.** La LSTM solo lee las primeras 200 palabras, y TF-IDF ve el correo completo. En los correos largos, la regresión logística tiene más información.
- **Sin ajuste.** La LSTM usa valores habituales (64, 0,5, 32, 6 épocas) en una sola ejecución, y la diferencia de AUC es pequeña (0,004).

### Qué ofrece la LSTM en principio

- **Orden y contexto.** "We will never ask you to confirm your password" (un aviso legítimo de seguridad) y "please confirm your password" (un fraude) comparten las palabras clave; lo que cambia el sentido es el orden y la negación. Un modelo que solo ve qué palabras aparecen lo tiene más difícil (criterio técnico; no lo medimos).
- **Fraudes bien escritos.** El correo de soporte TI se parece en vocabulario a un correo legítimo. Si algo delata a ese tipo de fraude, está más en cómo se combinan las frases que en palabras sueltas, y ahí el orden debería pesar más. Nuestra LSTM tampoco lo detectó, así que esto queda como hipótesis.

### Qué experimento lo comprobaría (no lo hicimos)

- Una prueba donde el orden importe: pares de correos con las mismas palabras y distinto sentido (por ejemplo, con negaciones), medidos con los dos modelos.
- Más fraudes bien escritos, como el de soporte TI, comparando cuántos detecta cada modelo.
- Más datos, más recientes y en español, para que la red tenga más ejemplos con que aprender su Embedding.
- Repetir el entrenamiento con varias semillas, para saber si la diferencia de AUC (0,991 contra 0,995) se mantiene o es variación entre ejecuciones.

### Por qué mostrar el empate es una fortaleza

- El enunciado pide justificar técnicamente la elección del modelo, y evalúa la comprensión y la argumentación, no solo el resultado. Comparar con una línea base es lo correcto; esconder el empate sería lo débil.
- La evaluación tiene "énfasis en Deep Learning", así que la solución debía usar un modelo de Deep Learning. La regresión logística es la vara que dice cuánto aporta: elegimos la LSTM por una razón técnica (el orden), la medimos con honestidad y reportamos lo que salió.
- Le deja a la organización una conclusión útil: con datos como estos, un modelo simple rinde igual; la LSTM se justificaría si los fraudes dependen más del contexto, y eso es lo que habría que medir.

### Respuesta corta para la defensa

"Lo medimos: en este dataset empatan, con 0,967 de exactitud los dos, y la regresión logística tiene un AUC un poco mayor, 0,995 contra 0,991. Pasa porque aquí las palabras sueltas ya separan casi todo, y la LSTM además solo lee 200 palabras. Elegimos la LSTM porque puede considerar el orden, que es lo que distinguiría a los fraudes bien escritos; para demostrarlo haría falta una prueba con ese tipo de correos. Preferimos mostrar el empate antes que decir que la red gana."

---

## 22. Cifras para tener a mano

Todas salen del notebook de respaldo, ejecutado en CPU local; en la grabación pueden variar un poco.

| Cifra | Valor | Paso |
|---|---|---|
| Filas originales | 18.650 | 4 |
| Seguros y fraudulentos originales | 11.322 y 7.328 (cerca de 61% y 39%) | 4 |
| Textos vacíos | 16 | 4 |
| Duplicados exactos | 1.112 | 4 |
| Mediana de palabras por correo | 160 | 4 |
| El 90% de los correos tiene menos de | 729 palabras | 4 |
| Correos con la palabra "enron" | 20,3% de los seguros; 0,0% de los fraudulentos | 4 |
| Correos únicos después de limpiar | 17.103 (10.661 seguros y 6.442 fraudulentos) | 5 |
| Entrenamiento, validación y prueba | 11.629, 2.053 y 3.421 | 6 |
| Vocabulario y largo | 20.000 palabras y 200 por correo | 7 |
| Forma de los datos de entrenamiento | 11.629 × 200 | 7 |
| Regresión logística en la prueba | Exactitud 0,967; fraude: precisión 0,972, recall 0,940, F1 0,956 | 8 |
| AUC de la regresión logística | 0,995 | 13 |
| Parámetros | Embedding 1.280.000; LSTM 33.024; Dropout 0; Dense 65; total 1.313.089 | 9 |
| Tasa de aprendizaje de Adam | 0,001 (por defecto de Keras) | 10 |
| Actualizaciones por época | 364 (lotes de 32) | 11 |
| Tiempo en CPU | 17 a 18 s por época; poco más de 1 minuto en total | 11 |
| Pérdida de validación por época | 0,532 → 0,111 → 0,168 → 0,122 | 11 |
| Parada temprana | Se detuvo después de la época 4 y recuperó la 2 (val_accuracy 0,969) | 11 |
| LSTM en la prueba | Exactitud 0,967; seguro: precisión 0,967, recall 0,981; fraude: precisión 0,967, recall 0,945, F1 0,956 | 13 |
| AUC de la LSTM | 0,991 | 13 |
| Matriz de confusión de la LSTM | 2.091 seguros bien, 41 falsas alarmas, 71 fraudes no detectados, 1.218 fraudes detectados | 13 y 15 |
| Si respondiera siempre "seguro" | Cerca de 62% de exactitud (2.132 de 3.421) | 13 |
| Correos nuevos | Banco 99,1%; Paquete 97,8%; Soporte TI 4,1% (se escapó); Coordinación 0,4% | 14 |
| Detección y falsas alarmas | Detecta 94,5% del fraude; 1,9% de falsas alarmas | 15 |
| Umbrales | 0,3: 95,7% y 2,4%; 0,5: 94,5% y 1,9%; 0,7: 92,1% y 1,5%; 0,9: 85,9% y 0,8% | 16 |
| Modo defensa | Exactitud 0,967; la frase de ejemplo da 0,987 (FRAUDULENTO) | Modo defensa |
| Errores de la regresión logística (deducidos) | 77 fraudes no detectados y 35 falsas alarmas: 112 errores, igual que la LSTM | Cálculo con el paso 8 |
| Entropía cruzada de un fraude | Con p = 0,9 cuesta 0,11; con p = 0,1 cuesta 2,30 | Cálculo con la fórmula del paso 10 |

---

## 23. Limitaciones que reconocemos

- **Empate con el modelo simple.** La regresión logística rinde prácticamente igual: en este dataset, la LSTM no mostró una mejora medible.
- **Los datos.** Están en inglés, mezclan phishing con spam común, y cerca del 20% de los correos seguros menciona a una sola empresa (Enron). Quitamos las palabras que la delatan, pero muchos correos seguros siguen siendo correos de trabajo de esa misma empresa, y el modelo podría estar aprendiendo ese estilo (inferencia).
- **Fraudes bien escritos.** Deja pasar fraudes sin lenguaje de urgencia, como el correo de soporte TI de la prueba.
- **Correos automáticos legítimos.** La conclusión del notebook advierte que puede marcar como fraude correos legítimos automáticos, como una recuperación de contraseña. Las pruebas del notebook no lo reproducen: es un riesgo razonable, no una medición.
- **Fraudes actuales.** Pueden ser distintos a los de este dataset.
- **Lo que el modelo no ve.** Solo lee las primeras 200 palabras; no ve el remitente ni el dominio de los enlaces; pierde números y símbolos, como un monto a pagar.
- **Una sola ejecución.** No medimos cuánto varían las cifras entre ejecuciones, y por ahora solo se ejecutó en un computador local, en CPU, no en Colab.

---

## 24. Mejoras

Las cuatro primeras son las de la conclusión del notebook; las dos últimas salen de la sección 21.

- Entrenar con correos recientes y en español.
- Sumar señales como el remitente y el dominio real de los enlaces.
- Separar phishing de spam en clases distintas.
- Enviar lo sospechoso a cuarentena, con revisión humana, y cuidar la privacidad del contenido de los correos (Ley 21.719).
- Probar los dos modelos con correos donde el orden importe y con más fraudes bien escritos, para comprobar si la LSTM aporta.
- Repetir el entrenamiento con varias semillas para medir cuánto varían los resultados.

---

## 25. Comparación con el material del profesor

Revisamos el ejemplo de noticias falsas con LSTM (`aplicacion_IA_NoticiasFalsas.pdf`) y el de regresión logística para spam (`App_regresion_logistica.docx`). El proyecto sigue el mismo camino del curso: texto → números → Embedding → LSTM → Dropout → Dense con sigmoide, compilado con adam y binary_crossentropy, y una prueba con un dato nuevo. Estas son las diferencias y qué responder:

| Aspecto | Ejemplo del profesor | Nuestro proyecto | Si preguntan |
|---|---|---|---|
| Texto a números | `Tokenizer(num_words=5000)` + `pad_sequences(maxlen=500)` | `TextVectorization(max_tokens=20000, output_sequence_length=200)` | Hace lo mismo en una sola capa, que además queda guardada dentro del modelo para la defensa. |
| Vocabulario | 5.000 palabras | 20.000 | Valor habitual, no ajustado: incluye más palabras distintivas a cambio de más parámetros. No probamos 5.000. |
| Largo | 500 palabras; `pad_sequences` por defecto rellena y corta al inicio, así que lee las últimas 500 | 200; rellena y corta al final, así que lee las primeras 200 | 200 por la mediana de 160 palabras y por el tiempo de entrenamiento en vivo. La máscara hace que la LSTM ignore el relleno. |
| Capas LSTM | Dos apiladas: `LSTM(64, return_sequences=True)`, `Dropout(0.5)`, `LSTM(32)` | Una: `LSTM(64)`, `Dropout(0.5)` | Es más simple y entrena más rápido en vivo. Con el empate medido, no hay evidencia de que haga falta más capacidad. No probamos dos capas. |
| Validación | Valida con los mismos datos de prueba | Validación separada (2.053 correos) y prueba solo al final | La parada temprana decide con la validación; si validáramos con la prueba, la prueba dejaría de ser independiente. |
| Épocas y lote | 5 épocas fijas, lote de 64 | Máximo 6 con parada temprana, lote de 32 | El orden de magnitud es el mismo; la parada temprana elige la mejor época con datos. |
| Etiquetas | `LabelEncoder` | Comparación directa (`== 'Phishing Email'`) | LabelEncoder ordena alfabéticamente y dejaría "Phishing Email" = 0; así aseguramos que 1 sea fraude. |
| Evaluación | `model.evaluate` (exactitud) y una noticia nueva | Reporte de clasificación, AUC, matriz de confusión, comparación con un modelo simple y 4 correos nuevos | Va más allá del ejemplo, en la línea de "accuracy no siempre es suficiente" del material de regresión logística. |

**Ojo con una palabra.** En el ejemplo, el profesor imprime la exactitud (accuracy) como "Precisión del modelo", y el material de modelos de machine learning guarda la exactitud en una variable llamada `precision`. Nosotros usamos exactitud para el porcentaje de aciertos y precisión para la métrica de la clase fraude. Si el profesor dice "precisión" refiriéndose a los aciertos, aclarar: "eso es la exactitud, 0,967; la precisión de la clase fraude también salió 0,967".

**Los casos A y B del material de regresión logística.** El material pregunta qué error molesta más: que un spam llegue a la bandeja (caso A) o que un correo legítimo se vaya a spam (caso B). En nuestro proyecto, el caso A es el falso negativo, el error caro, y el caso B es el falso positivo.

**Si preguntan por Isolation Forest** (el material lo asocia a la detección de fraude): es no supervisado y sirve cuando no hay etiquetas o el fraude es muy raro. Aquí tenemos etiquetas y el fraude es cerca del 38% de los correos, así que corresponde un modelo supervisado.
