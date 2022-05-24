# Clasificador de Heliconias

Colombia es el país con el mayor número de especies vegetales en el mundo. Dentro de ellas, las heliconias desempeñan un importante papel ecológico dentro de los ecosistemas, pues son componentes frecuentes del interior y límites de los bosques, así como de ambientes abiertos como potreros, bordes de carretera y orillas de ríos. En algunos ecosistemas actúan como pioneras en el proceso de regeneración natural de la vegetación y restauración del suelo degradado. Además, mantienen importantes relaciones coevolutivas con otras especies animales y vegetales, constituyéndose en un elemento importante dentro del complejo armazón de la vida en el trópico.

La clasificación de especies de plantas es crucial para la protección y conservación de la biodiversidad. La clasificación manual requiere mucho tiempo, es costosa y se necesitan expertos que a menudo tienen disponibilidad limitada. Para hacer frente a estos problemas, en este trabajo se utilizaron tres métodos de clasificación de imágenes SVM (Máquina de Vector de Soporte), ANN (Redes neuronales), KNN (Vecinos más próximos) con distancia euclidiana y de intersección, las cuales entregaron buenos resultados en la clasificación de cuatro especies de heliconias encontradas en la Universidad de Ibagué. 

Los datos empleados para el entrenamiento, prueba y validación de los métodos fueron imágenes RGB tomadas en el hábitat natural de las heliconias, con el fin de tener información desde su germinación hasta su momento óptimo de corte. Las imágenes fueron preprocesadas, haciendo un ajuste de balance de blancos, contraste y temperatura del color. Para separar las heliconias del fondo se utilizó una técnica de segmentación por grafos mediante SPG. Los descriptores se obtuvieron empleando la técnica conocida como BoW (Bag of Words), encontrando que el número de palabras visuales más adecuadas para la clasificación estaba entre 20 a 40. El método con el que se obtuvieron los mejores resultados fue el KNN; empleando los tres vecinos más cercanos, con una precisión del 97%.

## Ejecucion del codigo

El programa esta compuesto de dos archivos `Principal.py` y `malla.py`

## Principal.py

- Lee las imagenes de cada variedad de heliconia y las reduce de tamaña usando la funcion `malla.matriz(alto,ancho,imagen)`.
- Como las imagenes estan pre-procesadas con un tono color negro de fondo, se procede apilan en una sola matriz usando la funcion `malla.acomodar(imagen_RGB)`.
- Se utiliza el metodo Bag of words el cual acomoda los centroides seleccionados les asigna una etiqueta y realiza el conteo de esta etiqueta usando la fucnion `malla.codigo(imagens,cluster)`
- Se ingresa una matriz de colores y se procede a separa cada muestra para entrenamiento (60%), prueba (20%) y validacion (20%), tambien se retiran de la muestra el color negro ya que no aporta informacion relevante al procesamiento usando las siguientes funciones:
  - `malla.entrenamiento(im[0],im[1],im[2],im[3],labels,centroide)`
  - `malla.validacion(im[0],im[1],im[2],im[3],centroide)` 
  - `malla.prueba(im[0],im[1],im[2],im[3],centroide)`

- Por ultimo se ejecutan los clasificadores con su debida prueba y validacion.
  - `malla.KNN2()`
  - `malla.svm1()`
  - `malla.svm2()`
  - `malla.svm3()`
  - `malla.svm4()`
  - `malla.ANN()`
  - `malla.intersecion()`
  - `malla.intersecion_prom()`

## malla.py

Es una libreria que contiene las siguientes funciones

  - `read()` Lee las imagenes
  - `matriz()` Redimensiona las imagenes leidas y las apila
  - `matriz_test()` Redimensiona las imagenes leidas y las apila
  - `contar()` Cuenta la cantidad de datos que esta mas cerca al centroide
  - `acomodar()` Combierte de una matriz RGB a una matriz Nx3
  - `km()` Implementa la funcion K-means
  - `quitar_negros()` Retira todos los valores de color negro de la imagen
  - `graficar()` Realiza un grafico de barras
  - `codigo()` Combina las funciones `km()` y `contar()`
  - `KNN2()` Clasificador K-Nearest Neighbor (KNN) con diferente cantidad de vecinos
  - `KNN()` Se uso para agrupar los datos mas cercanos a los centroides
  - `svm#()` Clasificador Support Vector Machine (SVM) con diferentes parametros y rangos
  - `ANN()` Clasificador Red Neuronal Artificial (ANN) variando cantidad de neuronas y capas
  - `entrenamiento()` Utiliza el 60% de los datos para crear la base de datos
  - `validacion()` Utiliza el 20% de los datos para realizar la validacion de los datos
  - `prueba()` Utiliza el 20% de los datos para realizar la prueba de los datos
  - `()` 
  - `()` 

## Autor

- Cristian González (<cristian-saul-66@hotmail.com>)

## Publicación

- [Automatic Classification of Zingiberales from RGB Images](https://link.springer.com/chapter/10.1007/978-3-030-77004-4_19)
