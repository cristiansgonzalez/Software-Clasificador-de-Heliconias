# Clasificador de Heliconias

Colombia es el país con el mayor número de especies vegetales en el mundo. Dentro de ellas, las heliconias desempeñan un importante papel ecológico dentro de los ecosistemas, pues son componentes frecuentes del interior y límites de los bosques, así como de ambientes abiertos como potreros, bordes de carretera y orillas de ríos. En algunos ecosistemas actúan como pioneras en el proceso de regeneración natural de la vegetación y restauración del suelo degradado. Además, mantienen importantes relaciones coevolutivas con otras especies animales y vegetales, constituyéndose en un elemento importante dentro del complejo armazón de la vida en el trópico.

La clasificación de especies de plantas es crucial para la protección y conservación de la biodiversidad. La clasificación manual requiere mucho tiempo, es costosa y se necesitan expertos que a menudo tienen disponibilidad limitada. Para hacer frente a estos problemas, en este trabajo se utilizaron tres métodos de clasificación de imágenes SVM (Máquina de Vector de Soporte), ANN (Redes neuronales), KNN (Vecinos más próximos) con distancia euclidiana y de intersección, las cuales entregaron buenos resultados en la clasificación de cuatro especies de heliconias encontradas en la Universidad de Ibagué. 

Los datos empleados para el entrenamiento, prueba y validación de los métodos fueron imágenes RGB tomadas en el hábitat natural de las heliconias, con el fin de tener información desde su germinación hasta su momento óptimo de corte. Las imágenes fueron preprocesadas, haciendo un ajuste de balance de blancos, contraste y temperatura del color. Para separar las heliconias del fondo se utilizó una técnica de segmentación por grafos mediante SPG. Los descriptores se obtuvieron empleando la técnica conocida como BoW (Bag of Words), encontrando que el número de palabras visuales más adecuadas para la clasificación estaba entre 20 a 40. El método con el que se obtuvieron los mejores resultados fue el KNN; empleando los tres vecinos más cercanos, con una precisión del 97%.

## Autor

- Cristian González (<cristian-saul-66@hotmail.com>)

## Publicación

- [Automatic Classification of Zingiberales from RGB Images]([https://pages.github.com/](https://link.springer.com/chapter/10.1007/978-3-030-77004-4_19))
