# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 3: Reconocimiento de escenas con modelos BOW/BOF
# Tarea 2: extraccion de caracteristicas

# AUTOR1: APELLIDO1 APELLIDO1, NOMBRE1
# AUTOR2: APELLIDO2 APELLIDO2, NOMBRE2
# PAREJA/TURNO: NUMERO_PAREJA/NUMERO_TURNO

# librerias y paquetes por defecto
import numpy as np
from skimage import io, color, transform, feature
from sklearn.preprocessing import normalize
from p3_tests import test_p3_tarea2

def obtener_features_tiny(path_imagenes, tamano=16):
    """
    Calcula un descriptor basado en submuestreo para una lista de imágenes.

    Argumentos de entrada:
    - path_imagenes: Lista de strings, rutas de las imágenes.
    - tamano: Tamaño de la dimensión de cada imagen resultante después del redimensionado.
    
    Argumentos de salida:
    - list_img_desc_tiny: Lista 1xN, donde cada posición representa los descriptores calculados para cada imagen.
                          En el caso de características Tiny, cada posición contiene UN DESCRIPTOR 
                          con dimensiones 1xD donde D es el número de dimensiones del vector de características/feature Tiny.
                          Ejemplo: si tamano=16, entonces D = 16 * 16 = 256 y el vector será 1x256.
    """
    # Iniciar variable de salida
    list_img_desc_tiny = []

    for path in path_imagenes:
        # Cargar la imagen en escala de grises y en formato float
        img = io.imread(path)
        if len(img.shape) == 3:
            img = color.rgb2gray(img)
        img = transform.resize(img, (tamano, tamano))

        # Convertir la imagen en un vector fila
        img_flat = np.reshape(img, (1, -1))

        # Almacenar el descriptor en la lista de resultados
        list_img_desc_tiny.append(img_flat)
    
    return list_img_desc_tiny

def obtener_features_hog(path_imagenes, tamano=100, orientaciones=9, pixeles_por_celda=(8, 8), celdas_bloque=(2, 2)):
    """
    Calcula un descriptor basado en Histograma de Gradientes Orientados (HOG) para una lista de imágenes.

    Argumentos de entrada:
    - path_imagenes: Lista de strings, rutas de las imágenes.
    - tamano: Tamaño de la dimensión de cada imagen resultante tras aplicar el redimensionado.
    - orientaciones: Número de orientaciones a considerar en el descriptor HOG.
    - pixeles_por_celda: Tupla de int, número de píxeles en cada celda del descriptor HOG.
    - celdas_bloque: Tupla de int, número de celdas a considerar en cada bloque del descriptor HOG.

    Argumentos de salida:
    - list_img_desc_hog: Lista 1xN, donde cada posición representa los descriptores calculados para cada imagen.
                        En el caso de características HOG, cada posición contiene VARIOS DESCRIPTORES 
                        con dimensiones MxD donde 
                        - M es el número de vectores de características/features de cada imagen 
                        - D el número de dimensiones del vector de características/feature HOG.
                        Ejemplo: Para una imagen de 100x100 y con valores por defecto, 
                        para cada imagen se obtienen M=81 vectores/descriptores de D=144 dimensiones.
    """
    list_img_desc_hog = list()

    for path in path_imagenes:
         # Cargar la imagen en escala de grises y en formato float
        img = io.imread(path)
        if len(img.shape) == 3:
            img = color.rgb2gray(img)
        img = transform.resize(img, (tamano, tamano))
        
        # Calcular HOG para la imagen redimensionada
        hog_features = feature.hog(img, orientations=orientaciones,
                                   pixels_per_cell=pixeles_por_celda,
                                   cells_per_block=celdas_bloque,
                                   feature_vector=False)

        # Aplanar la matriz de características HOG
        hog_features_flattened = hog_features.reshape(hog_features.shape[0]*hog_features.shape[1], -1)
        # Agregar los descriptores a la lista
        list_img_desc_hog.append(hog_features_flattened)
    return list_img_desc_hog

if __name__ == "__main__":
    dataset_path = './dataset/scenes15'    
    print("Practica 3 - Tarea 1 - Test autoevaluación\n")                    
    print("Tests completados = " + str(test_p3_tarea2(dataset_path,stop_at_error=True,debug=True))) #analizar todos los casos sin pararse en errores ni mostrar datos
    #print("Tests completados = " + str(test_p3_tarea1(dataset_path,stop_at_error=True,debug=True))) #analizar todos los casos, pararse en errores y mostrar datos