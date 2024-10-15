import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re

noticias = [
    "El equipo ganó el partido de fútbol ayer", 
    "La política económica del país está cambiando", 
    "Nueva película de acción en cines este viernes", 
    "La reforma de ley fue aprobada en el Senado", 
    "Los actores se presentaron en la gala del cine", 
    "La selección nacional jugará la final", 
    "El presidente anunció nuevas medidas económicas en la conferencia", 
    "Se confirmó la fecha de lanzamiento del nuevo teléfono inteligente", 
    "El cantante principal del grupo ganó un premio en la gala", 
    "El equipo de baloncesto logró su tercera victoria consecutiva", 
    "Se revelaron avances en la investigación del cambio climático", 
    "Los desarrolladores de videojuegos anunciaron nuevas actualizaciones", 
    "La bolsa de valores cerró en números rojos por tercera semana consecutiva", 
    "La película de ciencia ficción superó todas las expectativas en taquilla", 
    "El congreso discute la aprobación de la nueva ley de salud", 
    "La liga de fútbol presentará cambios en su formato la próxima temporada", 
    "El parlamento votará mañana sobre la propuesta de reforma energética", 
    "El actor protagonizará una nueva serie dramática el próximo año", 
    "Las nuevas consolas de videojuegos han batido récords de ventas", 
    "El equipo de tenis se clasifica para la final del torneo", 
    "Se espera una nueva ola de protestas en la capital por las reformas", 
    "La cantante pop lanzará su nuevo álbum a finales de este mes", 
    "Los científicos descubren una nueva especie de animal en la selva", 
    "La ley de protección al medio ambiente fue aprobada en el congreso", 
    "El director de cine revela detalles de su próximo proyecto cinematográfico", 
    "El equipo de natación bate el récord mundial en los Juegos Olímpicos", 
    "El primer ministro anunció una reestructuración de su gabinete", 
    "El sistema operativo más reciente promete mejoras de seguridad significativas", 
    "El sector automotriz reporta un aumento en la producción de vehículos eléctricos", 
    "Los nuevos avances tecnológicos revolucionan la industria médica", 
    "La película animada se convierte en un éxito entre niños y adultos", 
    "El gobierno discutirá un plan para mejorar la educación pública", 
    "La actriz ganadora del Oscar protagonizará una comedia romántica", 
    "La empresa tecnológica presentó su nuevo software de inteligencia artificial", 
    "El equipo de ciclismo se prepara para la competencia más importante del año", 
    "Los ciudadanos se manifestaron en contra de la corrupción gubernamental", 
    "El festival de música atraerá a miles de fanáticos este fin de semana", 
    "El descubrimiento de un tratamiento para una enfermedad rara abre nuevas esperanzas", 
    "La economía mundial se enfrenta a nuevos desafíos debido a la pandemia", 
    "La cantante de pop lanzó un sencillo sorpresa que está arrasando en las listas", 
    "El parlamento aprueba una nueva ley de privacidad de datos", 
    "Los robots avanzan en la automatización de procesos en la industria", 
    "El equipo de rugby vence a su rival en un partido histórico", 
    "Los mercados globales sufren una caída después de las elecciones en el país", 
    "La banda de rock anuncia su última gira antes de su retiro definitivo"
]

categorias = [
    "deportes", "política", "entretenimiento", "política", "entretenimiento", "deportes", 
    "política", "tecnología", "entretenimiento", "deportes", "ciencia", "tecnología", 
    "economía", "entretenimiento", "política", "deportes", "política", "entretenimiento", 
    "tecnología", "deportes", "política", "entretenimiento", "ciencia", "política", 
    "entretenimiento", "deportes", "política", "tecnología", "economía", "tecnología", 
    "entretenimiento", "política", "entretenimiento", "tecnología", "deportes", 
    "política", "entretenimiento", "ciencia", "economía", "entretenimiento", 
    "política", "tecnología", "deportes", "economía", "entretenimiento"
]

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'\W+',' ',texto)
    return texto

noticias_limpias = [limpiar_texto(n) for n in noticias]
    

#Convertir el texto en vectores de caracteristicas usando TF-IDF
vectorirador = TfidfVectorizer()
x = vectorirador.fit_transform(noticias_limpias) 

#Dividir los datos entre entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, categorias, test_size=0.2, random_state=42)

#Crear modelo Naive Bayes
modelo = MultinomialNB()
modelo.fit(x_train,y_train)

#Predecir en datos de prueba
predicciones = modelo.predict(x_test)

#Calcular la precision del modelo
precision = accuracy_score(y_test,predicciones)
print(f"Presicion del clasificador de noticias : {precision:.2f}")

#Prediccion de ejemplo 
nueva_noticia = ["Se confirmó la fecha de lanzamiento del nuevo teléfono inteligente"]
nueva_noticia_vec = vectorirador.transform(nueva_noticia)
categoria_predicha = modelo.predict(nueva_noticia_vec)
print(f"La noticia se clasifica como : {categoria_predicha[0]}")