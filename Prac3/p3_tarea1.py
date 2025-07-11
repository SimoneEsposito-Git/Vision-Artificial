# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 3: Reconocimiento de escenas con modelos BOW/BOF
# Tarea 1: modelo BOW/BOF

# AUTOR1: APELLIDO1 APELLIDO1, NOMBRE1
# AUTOR2: APELLIDO2 APELLIDO2, NOMBRE2
# PAREJA/TURNO: NUMERO_PAREJA/NUMERO_TURNO

# librerias y paquetes por defecto
from p3_tests import test_p3_tarea1
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def construir_vocabulario(list_img_desc, vocab_size=5, max_iter=300):
    """   
    Construir el vocabulario utilizando K-Means para agrupar los descriptores.

    Argumentos de entrada:
    - list_img_desc: Lista 1xN con los descriptores de cada imagen.
    - vocab_size: Número de palabras para el vocabulario a construir.
    - max_iter: Número máximo de iteraciones del algoritmo KMeans.

    Argumentos de salida:
    - vocabulario: Numpy array de tamaño [vocab_size, D], 
                   que contiene los centros de los clusters obtenidos por K-Means.
    """
    # Concatenar todos los descriptores en una matriz única
    all_desc = np.concatenate(list_img_desc, axis=0)

    # Aplicar k-means clustering
    kmeans = KMeans(n_clusters=vocab_size, max_iter=max_iter, random_state=0)
    kmeans.fit(all_desc)

    # Obtener los centroides como el vocabulario
    vocabulario = kmeans.cluster_centers_

    return vocabulario

def obtener_bags_of_words(list_img_desc, vocab):
    """
    # Esta funcion obtiene el Histograma Bag of Words para cada imagen
    #
    # Argumentos de entrada:
    # - list_img_desc: Lista 1xN con los descriptores de cada imagen. Cada posicion de la lista 
    #                   contiene (MxD) numpy arrays que representan UNO O VARIOS DESCRIPTORES 
    #                   extraidos sobre la imagen
    #                   - M es el numero de vectores de caracteristicas/features de cada imagen 
    #                   - D el numero de dimensiones del vector de caracteristicas/feature.  
    #   - vocab: Numpy array de tamaño [vocab_size, D], 
    #                  que contiene los centros de los clusters obtenidos por K-Means.   
    #
    # Argumentos de salida: 
    #   - list_img_bow: Array de Numpy [N x vocab_size], donde cada posicion contiene 
    #                   el histograma bag-of-words construido para cada imagen.
    #
    """
   # iniciamos la variable de salida (numpy array)
    list_img_bow = np.empty(shape=[len(list_img_desc), len(vocab)])

    for i, img_desc in enumerate(list_img_desc):
        # Calcular las distancias entre los descriptores de la imagen y los centros del vocabulario
        distances = cdist(img_desc, vocab, 'euclidean')

        # Encontrar el índice del centro más cercano para cada descriptor
        closest_center_indices = np.argmin(distances, axis=1)

        # Construir el histograma Bag of Words para la imagen
        histogram = np.bincount(closest_center_indices, minlength=len(vocab))

        # Normalizar el histograma (opcional)
        histogram = histogram / np.sum(histogram)

        # Asignar el histograma a la lista de resultados
        list_img_bow[i, :] = histogram

    return list_img_bow
    
if __name__ == "__main__":
    dataset_path = 'C:\\Users\\konst\\Dokumente\\Uni_Module\\UAM\\CompVision\\Programming\\VisArt\\Prac3\\dataset\\scenes15\\'
    print("Practica 3 - Tarea 1 - Test autoevaluación\n")                    
    print("Tests completados = " + str(test_p3_tarea1(dataset_path,stop_at_error=False,debug=False))) #analizar todos los casos sin pararse en errores ni mostrar datos
    #print("Tests completados = " + str(test_p3_tarea1(dataset_path,stop_at_error=True,debug=True))) #analizar todos los casos, pararse en errores y mostrar datos
    #hurensohn