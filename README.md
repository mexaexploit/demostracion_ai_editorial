Clasificador de Noticias con Naive Bayes y TF-IDF
Este proyecto implementa un clasificador de noticias utilizando el modelo de Naive Bayes Multinomial y la técnica de TF-IDF para convertir los textos en vectores de características. El objetivo del modelo es clasificar noticias en diferentes categorías, como "Deportes", "Política", "Tecnología", "Entretenimiento", etc.

Descripción
El modelo está entrenado con un conjunto de datos que contiene títulos de noticias en varias categorías. La técnica de TF-IDF (Term Frequency - Inverse Document Frequency) se utiliza para transformar el texto en características numéricas que luego se utilizan para entrenar un modelo de Naive Bayes.

Además, se implementa un ejemplo de predicción sobre una nueva noticia, demostrando cómo el modelo puede generalizar a noticias no vistas previamente.

Tecnologías Utilizadas
Python: Lenguaje de programación principal.
scikit-learn: Utilizado para el modelo de Naive Bayes, la métrica de precisión y la partición de los datos.
NumPy: Biblioteca para operaciones numéricas.
re (expresiones regulares): Para la limpieza de los datos textuales.
Estructura del Proyecto
El proyecto está estructurado en los siguientes pasos:

Carga y limpieza de datos: Las noticias se preprocesan eliminando caracteres especiales y convirtiéndolas a minúsculas.
Conversión de texto en vectores TF-IDF: El texto se convierte en vectores de características utilizando TfidfVectorizer.
Entrenamiento y validación del modelo: Se entrena un modelo de Naive Bayes Multinomial y se evalúa su precisión.
Predicción en noticias nuevas: Se realiza una predicción sobre una noticia de ejemplo no vista previamente.

Requisitos
Asegúrate de tener instaladas las siguientes dependencias antes de ejecutar el código:

bash
Copy code
pip install numpy scikit-learn
Ejecución del Proyecto
Clona este repositorio:

bash
Copy code
git clone https://github.com/mexaexploit/demostracion_ai_editorial.git
Navega al directorio del proyecto:

bash
Copy code
cd demostracion_ai_editorial
Ejecuta el script principal para entrenar el modelo y ver los resultados:

bash
Copy code
python demostracion_ai_editorial.py
Para predecir la categoría de una noticia nueva, modifica el bloque de predicción de ejemplo en el script principal.
