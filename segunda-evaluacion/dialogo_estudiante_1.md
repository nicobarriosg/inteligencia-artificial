# Diálogo del estudiante 1: problema y modelo

Guion hablado de [Nombre estudiante 1] para el video, lámina por lámina. Cubre los roles de integrante 1 (el problema) e integrante 2 (el modelo). Después, [Nombre estudiante 2] programa la solución en vivo con `guia_phishing_en_vivo.md`.

## Cómo usar este diálogo

- Ensayar en voz alta 2 o 3 veces, con cronómetro. Las cifras toman más tiempo al decirlas de lo que sugiere el conteo de palabras.
- En el video, hablar desde las ideas en **negrita**, no desde el texto: la rúbrica castiga leer. El resto es una forma posible de decirlo, no un libreto para memorizar.
- Las láminas muestran solo palabras clave. Lo que va en cursiva y entre paréntesis es una acción, no se dice.
- Las frases marcadas [opcional] se pueden saltar si el tiempo se alarga.
- Tiempo objetivo: 13:30, que es el reloj de cada lámina (el máximo es 15:00). El texto completo da cerca de 14:00 a 125 palabras por minuto; sin las frases [opcional], cerca de 13:25. No adelantar resultados del modelo: los muestra [Nombre estudiante 2] en vivo.

---

## Lámina 1 · Detección de correos fraudulentos con una red LSTM · 0:00–0:20

*(en cámara, con la portada en pantalla)*

Hola, soy [Nombre estudiante 1]. Nuestro proyecto detecta correos fraudulentos con una red LSTM. Somos un grupo de dos, así que **yo voy a presentar el problema y el modelo**. Después, [Nombre estudiante 2] va a programar la solución en vivo.

## Lámina 2 · ¿Qué es el phishing? · 0:20–1:20

El correo es una herramienta central de trabajo, y también una de las puertas de entrada más comunes de los ataques. **El phishing es un correo que imita a un banco, a un servicio de despacho o al propio soporte de TI.** Busca que la persona entregue su clave, sus datos o dinero. Y llega mezclado con correos legítimos y con spam.

*(señala el correo de la lámina)* Este correo fue escrito para probar el proyecto. **Tiene cuatro señales típicas.** *(señala el 1)* Primero, la urgencia: pide actuar dentro de 24 horas. *(señala el 2)* Segundo, la amenaza de suspender la cuenta. *(señala el 3)* Tercero, un enlace falso. [opcional] Su dominio, punto invalid, ni siquiera existe. *(señala el 4)* Y cuarto, pide verificar datos personales. Al final, [Nombre estudiante 2] va a probar este mismo correo con el modelo.

## Lámina 3 · El problema que queremos resolver · 1:20–2:10

Ahora, el problema concreto. No hablamos de ciberseguridad en general. **Queremos clasificar cada correo que llega como seguro o fraudulento, a partir de su texto.** Y además, estimar qué tan probable es que sea fraude. *(sigue el flujo con la mano)* El correo entra al modelo y sale una probabilidad. Si es alta, el correo va a cuarentena y el área de TI lo revisa. Si es baja, llega a la bandeja de entrada. La organización puede ser una empresa o una universidad. Y esto es lo que necesita: **un filtro de apoyo, con revisión humana**. El filtro no borra nada por su cuenta.

## Lámina 4 · ¿A quién afecta y qué pasa si no hacemos nada? · 2:10–3:20

¿A quién afecta? *(señala los íconos)* A las personas que reciben los correos, sean trabajadores o estudiantes. Al área de TI, que revisa los reportes a mano. A la organización, con sus sistemas y su reputación. Y a los clientes, cuyos datos guarda la organización.

¿Y si no hacemos nada? *(sigue la cadena de consecuencias)* **Basta con que una sola persona caiga.** Con su clave, el atacante entra a la cuenta. Desde ahí puede robar datos o dinero, o atacar a otros desde adentro. Y revisar todo a mano no escala.

*(señala las leyes)* Y hay un lado legal. En Chile están la Ley 21.663, marco de ciberseguridad, y la Ley 21.719, de protección de datos personales.

*(señala las cifras)* Según la Fiscalía de Chile, en 2025 el 22% de los delitos ingresados fueron uso malicioso de tarjetas o claves, o estafas. Y la Fiscalía nombra el phishing entre las variantes de este fraude. [opcional] Además, según la Agencia Nacional de Ciberseguridad, al cierre de 2025 había 3.258 instituciones inscritas para reportarle incidentes.

## Lámina 5 · Los datos para estudiar el problema · 3:20–4:15

Para estudiar el problema usamos un dataset público de Kaggle, Phishing Email Detection. *(señala los números)* Tiene 18.650 correos: 11.322 seguros y 7.328 fraudulentos. Más o menos 6 de cada 10 son seguros.

**Cada correo trae su etiqueta, seguro o fraudulento, y por eso es aprendizaje supervisado.** El modelo aprende la relación entre el texto y su etiqueta. Sin etiquetas, sería aprendizaje no supervisado: por ejemplo, agrupar correos parecidos sin saber cuáles son fraude.

*(señala las tres limitaciones)* Los datos tienen límites. Están en inglés. Mezclan phishing con spam común. Y uno de cada cinco correos seguros menciona a Enron, una sola empresa. [opcional] El modelo podría aprender un atajo: si dice Enron, es seguro. [Nombre estudiante 2] va a mostrar cómo se trata eso al limpiar el texto.

## Lámina 6 · ¿Por qué usar Inteligencia Artificial? · 4:15–5:00

¿Por qué Inteligencia Artificial y no reglas fijas? *(señala la columna izquierda)* Una regla fija sería una lista de palabras prohibidas o un filtro por remitente. El problema es que **el atacante cambia una palabra y pasa**. *(señala la columna derecha)* Un modelo, en cambio, aprende de miles de ejemplos. Reconoce patrones como la urgencia, las amenazas, los pedidos de clave o los enlaces. Y no entrega solo un sí o un no: **entrega una probabilidad**. Con ese número, la organización decide cuánto riesgo acepta. [opcional] Y si los ataques cambian, el modelo se puede volver a entrenar con correos nuevos.

## Lámina 7 · Oportunidades, beneficios y riesgos · 5:00–6:00

*(señala Oportunidades)* Esto abre oportunidades: un filtro de apoyo para TI, o un servicio para pymes, colegios o municipios sin equipo de seguridad. [opcional] Y los correos marcados sirven para capacitar.

*(señala Beneficios)* ¿Beneficios? Menos fraudes en la bandeja, menos revisión manual y una decisión medible.

*(señala Riesgos)* Y hay riesgos. Un fraude que se escapa es el error más grave. Muchas falsas alarmas quitan confianza. Los ataques cambian. Y hay un tema ético: el modelo lee correos que pueden tener datos personales. **Por eso hay que cuidar la privacidad y mantener la revisión humana.**

*(señala la frase de abajo)* Entonces, ¿qué problema queremos resolver? **Queremos detener el phishing antes de que alguien caiga**, clasificando cada correo por su texto. Y es importante porque basta una persona para poner en riesgo a la organización. Para eso necesitamos un modelo que entienda texto.

## Lámina 8 · ¿Por qué una red LSTM? · 6:00–7:00

*(pausa breve: empieza la parte del modelo)* Ahora, el modelo. Elegimos una red LSTM, del inglés Long Short-Term Memory, que es una red neuronal recurrente. El problema es de clasificación binaria de texto, con aprendizaje supervisado. Entra un correo y sale la probabilidad de que sea fraude.

¿Por qué esta red? Porque **un correo es una secuencia, y el orden de las palabras importa**. *(señala la frase)* Por ejemplo, "click the link below to confirm your password". La LSTM la lee palabra por palabra, con memoria. *(señala la bolsa)* Un modelo de bolsa de palabras solo mira qué palabras aparecen. Si las desordeno, para él es el mismo correo.

Se basa en el ejemplo de noticias falsas visto en clases, pero con una sola LSTM en vez de dos. Además, [Nombre estudiante 2] va a entrenar una regresión logística, un modelo simple de bolsa de palabras. **Así comprobamos si la LSTM realmente aporta.**

## Lámina 9 · Del texto a números · 7:00–7:50

¿Cómo entra un correo a la red? **Una red no entiende letras, solo números.** *(sigue el flujo)* Primero se limpia el texto, y cada enlace se cambia por la palabra enlaceweb. Luego se arma un vocabulario de hasta 20.000 palabras, las más frecuentes. **Cada palabra se reemplaza por su número.** *(señala el ejemplo)* Fíjense: workshop aparece dos veces, y siempre es el 149. Las palabras fuera del vocabulario comparten un mismo código. Cada correo queda en 200 números. Si es más corto, se rellena con ceros, que la red ignora. Si es más largo, se corta. Como la mediana es de 160 palabras, más de la mitad cabe completa. [Nombre estudiante 2] va a mostrar este mismo ejemplo en vivo.

## Lámina 10 · La arquitectura del modelo · 7:50–8:45

*(recorre los bloques de izquierda a derecha)* Entran los 200 números del correo. La primera capa es el Embedding: una tabla que convierte cada número en un vector de 64 valores. **Esa tabla se aprende al entrenar, y las palabras que se usan en contextos parecidos terminan con vectores parecidos.** Además, avisa a las capas siguientes que los ceros son relleno. Después viene la LSTM, con 64 unidades. **Lee los vectores en orden y resume el correo en 64 valores.** Luego, el Dropout, que ayuda contra el sobreajuste. Y al final, una capa Dense de una neurona con sigmoide entrega la probabilidad de fraude. *(señala el número grande)* Son 1.313.089 parámetros, y el 97% está en el Embedding: 20.000 palabras por 64 valores.

## Lámina 11 · La neurona artificial · 8:45–9:30

Ahora, el marco teórico. Partamos por la pieza básica: **la neurona artificial**. *(señala las entradas)* Recibe varios valores y multiplica cada uno por un peso. **El peso dice cuánto importa esa entrada.** Luego suma todo y agrega un sesgo, que desplaza el resultado. *(señala la f)* Después aplica una función de activación, que transforma esa suma. En la salida usamos la sigmoide: deja cualquier número entre 0 y 1, y por eso la leemos como una probabilidad. Dentro de la LSTM hay sigmoides y también la tangente hiperbólica, que deja los valores entre menos 1 y 1. Entrenar la red es ajustar todos estos pesos y sesgos.

## Lámina 12 · Por dentro de la LSTM · 9:30–10:35

¿Cómo recuerda la LSTM? *(señala la línea de arriba)* Tiene una memoria, el estado de celda. En cada paso recibe la palabra actual y el estado anterior. *(señala cada compuerta)* Y **tres compuertas controlan la memoria**. La de olvido decide qué borrar. La de entrada, qué guardar de la palabra nueva. Y la de salida, qué parte de la memoria entregar. Las compuertas usan sigmoide, como una llave de paso entre 0 y 1. La tangente hiperbólica propone los valores nuevos. Al final, solo el último estado pasa a la capa siguiente.

¿Por qué no una red recurrente simple? Porque en secuencias largas su gradiente se desvanece, y olvida las palabras lejanas. **La LSTM actualiza su memoria sumando, y así recuerda por muchos pasos.** *(señala la nota)* ¿Y sus 33.024 parámetros? Son cuatro bloques de pesos: las tres compuertas y los valores nuevos. Cada bloque tiene 64 por 128 pesos, más 64 sesgos.

## Lámina 13 · ¿Cómo aprende? · 10:35–11:50

La red aprende en un ciclo de cuatro pasos. *(señala el paso 1)* Primero, la propagación hacia adelante: el correo pasa por las capas y sale una probabilidad, p. *(señala el paso 2)* Segundo, se calcula el error con la función de pérdida. Usamos la **entropía cruzada binaria**, que compara p con la etiqueta real. *(señala el ejemplo)* Si un correo es fraude y la red dice 0,9, la pérdida es 0,11. Si dice 0,1, sube a 2,30. **Castiga fuerte equivocarse con seguridad.** *(señala el paso 3)* Tercero, la retropropagación: calcula cuánto aportó cada peso al error, desde la salida hacia atrás. En la LSTM, también hacia atrás por las palabras. *(señala el paso 4)* Cuarto, el descenso del gradiente: mueve cada peso un poco en la dirección que reduce el error. Usamos Adam, que adapta el tamaño del paso de cada peso. *(señala el costado)* Esto se hace por lotes de 32 correos. Con 11.629 correos de entrenamiento, son 364 actualizaciones por época. Una época es una pasada completa por ellos, y el máximo es 6.

## Lámina 14 · Sobreajuste y métricas · 11:50–13:00

*(señala el esquema)* **El sobreajuste es cuando la red memoriza los correos de entrenamiento y falla con los nuevos.** Este gráfico es solo un esquema. La señal: la pérdida de entrenamiento baja, pero la de validación sube. Para evitarlo, el Dropout pone en cero al azar la mitad de los valores de la LSTM, solo al entrenar. [opcional] Así, la red no depende de unas pocas unidades. Además, separamos 2.053 correos de validación, que no se usan para ajustar pesos. Y con parada temprana, si la pérdida de validación no mejora en 2 épocas, se detiene y recupera los mejores pesos.

*(señala la matriz)* La exactitud sola engaña: decir siempre "seguro" ya acierta 6 de cada 10. En la matriz, **el error caro es el fraude que se escapa: el falso negativo**. Por eso miramos sobre todo el recall: de todos los fraudes, cuántos detecta. La precisión dice, de lo que marca, cuánto era fraude. Y el AUC mide qué tan bien separa las clases: 1 es perfecto y 0,5 es azar.

## Lámina 15 · Lo que entrega el modelo · 13:00–13:30

*(señala la barra)* La salida es una probabilidad de fraude. Desde 0,5, el correo va a cuarentena; bajo 0,5, a la bandeja. **Ese umbral lo decide la organización**: si se baja, detecta más fraude, pero con más falsas alarmas. Entonces, **la LSTM es adecuada porque el correo es una secuencia**. Lo lee en orden, con memoria, y aprende reduciendo su error. Ahora, [Nombre estudiante 2] la va a implementar en vivo.

---

## Si me quedo en blanco

- **Lámina 1:** me presento · problema y modelo · [Nombre estudiante 2] en vivo
- **Lámina 2:** puerta de entrada · imita · urgencia, amenaza, enlace, datos · se prueba al final
- **Lámina 3:** seguro o fraudulento · probabilidad · cuarentena · revisión humana
- **Lámina 4:** personas, TI, organización, clientes · una persona cae · no escala · leyes · Fiscalía
- **Lámina 5:** 18.650 correos · etiqueta = supervisado · sin etiqueta = no supervisado · inglés, spam, Enron
- **Lámina 6:** reglas que se esquivan · aprende de ejemplos · patrones · probabilidad
- **Lámina 7:** oportunidades · beneficios · riesgos · privacidad y revisión humana · detener antes de que caiga
- **Lámina 8:** LSTM recurrente · secuencia y orden · bolsa de palabras · regresión logística
- **Lámina 9:** limpieza · vocabulario de 20.000 · el 149 · 200 números · ceros y corte
- **Lámina 10:** Embedding · LSTM · Dropout · Dense con sigmoide · 97% en el Embedding
- **Lámina 11:** pesos · sesgo · activación · sigmoide de 0 a 1 · tanh
- **Lámina 12:** memoria · olvido, entrada, salida · gradiente que se desvanece · cuatro bloques
- **Lámina 13:** adelante · pérdida 0,11 y 2,30 · hacia atrás · Adam · lote de 32 y época
- **Lámina 14:** memoriza · Dropout, validación, parada temprana · falso negativo · recall, precisión, AUC
- **Lámina 15:** probabilidad · umbral de la organización · secuencia · paso a [Nombre estudiante 2]

## Tiempos

| Lámina | Palabras habladas | De ellas, [opcional] | Tiempo estimado (125 palabras/min) | Tiempo objetivo |
|---|---:|---:|---:|---:|
| 1 | 39 | 0 | 0:19 | 0:20 |
| 2 | 120 | 7 | 0:58 | 1:00 |
| 3 | 98 | 0 | 0:47 | 0:50 |
| 4 | 158 | 18 | 1:16 | 1:10 |
| 5 | 120 | 11 | 0:58 | 0:55 |
| 6 | 92 | 15 | 0:44 | 0:45 |
| 7 | 130 | 7 | 1:02 | 1:00 |
| 8 | 131 | 0 | 1:03 | 1:00 |
| 9 | 113 | 0 | 0:54 | 0:50 |
| 10 | 112 | 0 | 0:54 | 0:55 |
| 11 | 101 | 0 | 0:48 | 0:45 |
| 12 | 142 | 0 | 1:08 | 1:05 |
| 13 | 155 | 0 | 1:14 | 1:15 |
| 14 | 165 | 9 | 1:19 | 1:10 |
| 15 | 67 | 0 | 0:32 | 0:30 |
| **Total** | **1.743** | **67** | **13:57** | **13:30** |

Conteo: solo el texto hablado, sin títulos ni acotaciones, incluidas las frases [opcional]; cada nombre entre corchetes cuenta como 2 palabras. Sin las frases [opcional] quedan 1.676 palabras, cerca de 13:25. Las cifras tardan más en decirse de lo que indica el conteo: el tiempo real lo da el ensayo con cronómetro.
