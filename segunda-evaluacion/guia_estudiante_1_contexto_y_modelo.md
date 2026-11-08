# Guía del estudiante 1: problema y modelo (roles de integrantes 1 y 2)

Grupo de 2. El estudiante 1 cubre la contextualización del problema (integrante 1) y la fundamentación del modelo (integrante 2). El estudiante 2 programa en vivo con `guia_phishing_en_vivo.md`.

Apoyo: `presentacion_estudiante_1.pdf` (15 láminas) y `dialogo_estudiante_1.md` (texto hablado, lámina por lámina).

Tiempo máximo: **15 minutos** (límite por integrante). Reparto sugerido: **6 min problema + 8 min modelo + 1 min de margen**.

Estas son ideas en puntos, no un texto para leer. La rúbrica castiga la lectura: hay que decirlo con palabras propias. Todas las cifras salen del notebook `deteccion_phishing_lstm.ipynb`; si en la grabación salen otros números, el estudiante 2 usa los suyos.

---

## Parte A. El problema (~6 min)

Pregunta que debe quedar respondida: **¿qué problema queremos resolver y por qué importa?**

Empezar presentándose: nombre y "voy a presentar el problema y el modelo".

### 1. Contexto general
- El correo es el canal de trabajo principal de una organización y también una de las puertas de entrada más comunes de los ataques.
- Phishing: un correo que imita a un banco, a un servicio de despacho o al propio soporte TI para que la persona entregue su clave, sus datos o haga un pago.
- Llega mezclado con correos legítimos y con spam común.

### 2. La situación específica que resolvemos
- No es "ciberseguridad" en general. Es esto: **clasificar automáticamente cada correo entrante como seguro o fraudulento, a partir de su texto**, y estimar qué tan probable es que sea fraude.
- Con esa probabilidad, el área de TI puede enviar lo sospechoso a cuarentena y revisarlo antes de que llegue a la bandeja.

### 3. Quiénes se ven afectados
- Los trabajadores que reciben los correos (o los estudiantes, si la organización es una universidad).
- El área de TI, que hoy revisa reportes a mano.
- La organización completa: sus sistemas, sus datos y su reputación.
- Los clientes o personas cuyos datos personales administra la organización.

### 4. Consecuencias si no se hace nada
- Basta con que **una** persona caiga para que se comprometa una cuenta, y desde ahí se puede robar información o atacar a otros.
- Pérdidas de dinero (pagos falsos), filtración de datos, horas de TI dedicadas a revisar y a contener incidentes.
- Obligaciones legales: la Ley 21.663 (marco de ciberseguridad) y la Ley 21.719 (protección de datos personales) aumentan la responsabilidad de las organizaciones ante incidentes y filtraciones.
- Revisar a mano no escala: el volumen de correos crece y los ataques cambian.

### 5. Datos para estudiar el problema
- Dataset público de Kaggle "Phishing Email Detection": **18.650 correos** etiquetados como seguros (11.322) o fraudulentos (7.328).
- Es aprendizaje **supervisado**: cada correo trae la respuesta correcta, y el modelo aprende de esos ejemplos.
- Decirlo con honestidad: los correos están en inglés, mezclan phishing con spam, y muchos correos seguros vienen de una sola empresa (Enron). El estudiante 2 muestra cómo se trata eso.

### 6. Por qué la IA aporta valor
- Las reglas fijas (listas de palabras, filtros por remitente) se esquivan fácil: el atacante cambia una palabra.
- Un modelo aprende patrones a partir de miles de ejemplos: urgencia, amenazas, pedidos de clave, enlaces.
- Entrega una probabilidad, no solo un sí o un no, y eso permite decidir cuánto riesgo aceptar.

### 7. Oportunidades de mejora o de negocio
- Filtro de apoyo para el área de TI: prioriza qué revisar.
- Servicio para organizaciones sin equipo de seguridad propio (pymes, colegios, municipios).
- Base para capacitar: los correos que el modelo marca sirven como ejemplos reales para entrenar a las personas.

### 8. Amenazas, riesgos y dificultades
- Los atacantes cambian su forma de escribir; un modelo entrenado con correos antiguos puede quedar desactualizado.
- **Falsos negativos**: un fraude que pasa como seguro. Es el error más grave.
- **Falsas alarmas**: un correo legítimo que se bloquea; si son muchas, la gente deja de confiar en el filtro.
- Privacidad: el modelo lee el contenido de los correos, que puede tener datos personales.

### 9. Beneficios esperados
- Menos correos fraudulentos que llegan a las personas.
- Menos tiempo de TI revisando a mano.
- Una decisión medible: cuántos fraudes detecta y cuántas falsas alarmas genera.

**Frase de cierre de la parte A (con palabras propias):** queremos un filtro que lea el texto de cada correo y estime la probabilidad de que sea fraude, para que la organización detenga el phishing antes de que alguien caiga. Para eso necesitamos un modelo que entienda texto, y eso nos lleva al modelo.

---

## Parte B. El modelo (~8 min)

Pregunta que debe quedar respondida: **¿por qué este modelo es adecuado y cómo funciona?**

### 1. Qué modelo y qué tipo de problema
- **Red neuronal recurrente LSTM** (Long Short-Term Memory), con una capa de Embedding antes.
- Problema de **clasificación binaria de texto**: entra un correo y sale la probabilidad de que sea fraudulento (1) o seguro (0).

### 2. Por qué es adecuado
- Un correo es una **secuencia** de palabras, y el orden importa: "click the link below to confirm your password" significa algo distinto que las mismas palabras desordenadas.
- Una red normal recibe todo de una vez y no ve el orden. Una LSTM lee palabra por palabra y guarda una memoria de lo que ya leyó.
- Se basa en el ejemplo de noticias falsas visto en clases (Embedding de 64, LSTM, Dropout de 0,5 y salida sigmoide), pero con una sola LSTM en vez de dos apiladas: es más simple y entrena más rápido en vivo.
- **Ser honesto con la comparación** (el estudiante 2 lo va a mostrar): también se mide una regresión logística, un modelo simple. En este dataset rinden casi igual (exactitud 0,967 los dos). Lo que se justifica es probar la LSTM porque considera el orden; **no** decir que es mejor que el modelo clásico. La ventaja del orden debería verse con fraudes mejor escritos, que son justamente los que hoy se escapan.

### 3. Cómo entran los datos
- Una red no entiende letras. Cada correo se limpia (minúsculas, sin HTML, los enlaces se reemplazan por la palabra `enlaceweb`) y cada palabra se reemplaza por su número en un vocabulario de las **20.000** palabras más frecuentes.
- Cada correo queda como una lista de **200 números** (los cortos se rellenan con ceros, los largos se cortan). La mediana es 160 palabras, así que 200 cubre la mayoría.

### 4. Arquitectura, capa por capa
| Capa | Qué hace | Parámetros |
|---|---|---|
| Embedding (20.000 × 64) | Convierte cada número de palabra en un vector de 64 valores. Palabras que se usan parecido terminan con vectores parecidos. Ignora el relleno de ceros. | 1.280.000 |
| LSTM (64 unidades) | Lee los vectores en orden y resume todo el correo en 64 valores. | 33.024 |
| Dropout (0,5) | Solo al entrenar, pone en cero al azar la mitad de los valores que entrega la LSTM, para evitar que memorice. | 0 |
| Dense (1 neurona, sigmoide) | Convierte el resumen en una probabilidad entre 0 y 1. | 65 |

Total: **1.313.089 parámetros**, que la red ajusta durante el entrenamiento.

### 5. Marco teórico breve (explicar el significado, sin demostraciones)
- **Neurona artificial:** recibe valores, multiplica cada uno por un **peso**, suma un **sesgo** y aplica una función de activación. Los pesos dicen cuánto importa cada entrada.
- **Funciones de activación:**
  - **Sigmoide** en la salida: aplasta cualquier número a un valor entre 0 y 1, que se lee como probabilidad.
  - Dentro de la LSTM hay compuertas con sigmoide (deciden cuánto pasa, de 0 a 1) y **tanh** (valores entre -1 y 1).
- **Cómo recuerda la LSTM:** tiene una memoria (estado de celda) y tres compuertas:
  - de **olvido**: qué borrar de lo que ya leyó;
  - de **entrada**: qué guardar de la palabra nueva;
  - de **salida**: qué parte de la memoria entregar.
  Eso le permite relacionar palabras que están lejos en el correo, cosa que una red recurrente simple pierde.
- **Propagación hacia adelante:** el correo pasa por las capas y sale una probabilidad, por ejemplo 0,98.
- **Función de pérdida: entropía cruzada binaria** (binary cross-entropy). Mide qué tan lejos está la probabilidad de la respuesta real. Castiga mucho equivocarse con seguridad: decir 0,01 a un fraude cuesta mucho más que decir 0,4.
- **Retropropagación:** calcula cuánto contribuyó cada peso al error, desde la salida hacia atrás.
- **Descenso del gradiente (optimizador Adam):** mueve cada peso un poco en la dirección que reduce el error. Adam ajusta solo el tamaño del paso de cada peso.
- **Lote (batch) de 32:** los pesos se actualizan cada 32 correos, no con uno ni con todos. Con 11.629 correos de entrenamiento son 364 actualizaciones por época.
- **Época:** una pasada completa por todos los correos de entrenamiento. Se fijan máximo 6.
- **Sobreajuste:** cuando la red memoriza los correos de entrenamiento y falla con correos nuevos. Se controla con:
  - **Dropout**;
  - un conjunto de **validación** separado, que la red no usa para aprender;
  - **parada temprana** (early stopping): si el error en validación no mejora en 2 épocas, se detiene y recupera los mejores pesos.
- **Métricas:**
  - **Exactitud:** porcentaje de aciertos. No basta sola.
  - **Recall de fraude:** de todos los fraudes, cuántos detecta. Es la más importante, porque un fraude que pasa es el error caro.
  - **Precisión:** de lo que marca como fraude, cuánto lo era de verdad. Mide las falsas alarmas.
  - **AUC:** qué tan bien separa ambas clases con cualquier umbral (1 es perfecto, 0,5 es azar).

### 6. Qué salida esperamos
- Una probabilidad entre 0 y 1 para cada correo. Con umbral 0,5: sobre 0,5 se marca fraudulento.
- El umbral se puede mover: bajarlo detecta más fraude a cambio de más falsas alarmas. Esa decisión es de la organización, no del modelo.

**Frase de cierre de la parte B (con palabras propias):** elegimos una LSTM porque el correo es una secuencia y el orden de las palabras importa; aprende ajustando sus pesos para reducir la entropía cruzada, y entrega una probabilidad de fraude. Ahora [Nombre estudiante 2] la va a construir y entrenar en vivo.

---

## Preguntas probables en la defensa (estudiante 1)

| Pregunta | Idea de respuesta |
|---|---|
| ¿Por qué Deep Learning si la regresión logística empata? | Lo medimos y lo decimos. Elegimos la LSTM porque considera el orden; en este dataset las palabras sueltas ya separan casi todo, por eso empatan. Con fraudes mejor escritos, el orden debería pesar más. |
| ¿Supervisado o no supervisado? | Supervisado: cada correo trae su etiqueta, seguro o fraudulento. |
| ¿Qué es el Embedding? | Una tabla que convierte cada palabra en un vector de 64 números, aprendida en el entrenamiento. |
| ¿Qué diferencia una LSTM de una RNN simple? | Las compuertas y la memoria de celda, que le permiten recordar información de muchas palabras atrás. |
| ¿Por qué sigmoide y entropía cruzada binaria? | Problema de dos clases: la sigmoide da una probabilidad y la entropía cruzada binaria es la pérdida que corresponde a esa salida. |
| ¿Qué es más grave, un falso positivo o un falso negativo? | El falso negativo: un fraude que llega a la bandeja. Por eso miramos el recall. |
| ¿Cómo evitaron el sobreajuste? | Dropout, validación separada y parada temprana. |
| ¿Qué riesgos éticos tiene? | Lee contenido privado (Ley 21.719), puede bloquear correos legítimos, y no reemplaza la revisión humana. |
| ¿Por qué 200 palabras? | Cubre más que la mediana (160) y mantiene el entrenamiento rápido. Lo que viene después de la palabra 200 no se lee: es una limitación. |
