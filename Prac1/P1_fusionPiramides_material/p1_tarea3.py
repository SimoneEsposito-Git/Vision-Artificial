# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 1: Fusion de imagenes mediante piramides
# Tarea 3: fusion de piramides y reconstruccion

# AUTOR1: APELLIDO1 APELLIDO1, NOMBRE1
# AUTOR2: APELLIDO2 APELLIDO2, NOMBRE2
# PAREJA/TURNO: NUMERO_PAREJA/NUMERO_TURNO

import numpy as np
from p1_tests import test_p1_tarea3
from p1_tarea1 import expand

def fusionar_lapl_pyr(lapl_pyr_imgA, lapl_pyr_imgB, gaus_pyr_mask):
    """ 
    # Esta funcion realiza la fusion entre dos piramides laplacianas de distintas imagenes.
    #   La fusion esta determinada por una mascara de la cual se utiliza su correspondiente
    #   piramide Gaussiana para combinar las dos piramides laplacianas.
    #
    # Argumentos de entrada:
    #   lapl_pyr_imgA: lista de numpy arrays obtenida con la funcion 'lapl_piramide' sobre una imagen img2
    #   lapl_pyr_imgB: lista de numpy arrays obtenida con la funcion 'lapl_piramide' sobre otra imagen img1
    #   gaus_pyr_mask: lista de numpy arrays obtenida con la funcion 'gaus_piramide' 
    #                  sobre una mascara para combinar ambas imagenes. 
    #                  Esta mascara y la piramide tiene valores en el rango [0,1]
    #                  Para los pixeles donde gaus_pyr_mask==1, se coge la piramide de img1
    #                  Para los pixeles donde gaus_pyr_mask==0, se coge la piramide de img2
    #    
    # Devuelve:
    #   fusion_pyr: piramide fusionada
    #       lista de numpy arrays con variable tamaño con "niveles+1" elementos.    
    #       fusion_pyr[i] es el nivel i de la piramide que contiene bordes
    #       fusion_pyr[niveles] es una imagen (RGB o escala de grises)
    """ 
    fusion_pyr = [] # iniciamos la variable
    for i in range(len(lapl_pyr_imgA)):
        fusion_pyr.append(lapl_pyr_imgA[i] * gaus_pyr_mask[i] + lapl_pyr_imgB[i] * (1 - gaus_pyr_mask[i]))
    #...
    
    return fusion_pyr

def reconstruir_lapl_pyr(lapl_pyr):
    """ 
    # Esta funcion reconstruye la imagen dada una piramide laplaciana.
    #
    # Argumentos de entrada:
    #   lapl_pyr: lista de numpy arrays obtenida con la funcion 'lapl_piramide' sobre una imagen img
    #    
    # Devuelve:
    #   output: numpy array con dimensiones iguales al primer nivel de la piramide lapl_pyr[0]
    #
    # NOTA: algunas veces, la operacion 'expand' devuelve una imagen de tamaño mayor 
    # que el esperado. Entonces no coinciden las dimensiones de la imagen expandida 
    #   del nivel k+1 y las dimensiones del nivel k. Verifique si ocurre esta 
    #   situacion y en caso afirmativo, elimine los ultimos elementos de la 
    #   imagen expandida hasta coincidir las dimensiones del nivel k
    #   Por ejemplo, si el nivel tiene tamaño 5x7, tras aplicar 'reduce' y 'expand' 
    #   obtendremos una imagen de tamaño 6x8. En este caso, elimine la 6 fila y 8 
    #   columna para obtener una imagen de tamaño 5x7 donde pueda aplicar la resta
    """ 
    output = np.empty(shape=[0,0]) # iniciamos la variable de salida (numpy array)
    for i in range(len(lapl_pyr)-1,0, -1):
        expanded_level = expand(lapl_pyr[i])    
        rows_exp, col_exp = expanded_level.shape
        rows_org, col_org = lapl_pyr[i-1].shape

        if rows_exp != rows_org:
            expanded_level = np.delete(expanded_level, lapl_pyr[i-1].shape[0], axis= 0)

        if col_exp != col_org:
            expanded_level = np.delete(expanded_level, lapl_pyr[i-1].shape[1], axis = 1)
        lapl_pyr[i-1] += expanded_level
    output = lapl_pyr[0]
    return output
    """
    if len(lapl_pyr) <= 2:
        expanded_level = expand(lapl_pyr[1])    
        rows_exp, col_exp = expanded_level.shape
        rows_org, col_org = lapl_pyr[0].shape

        if rows_exp != rows_org:
            expanded_level = np.delete(expanded_level, lapl_pyr[0].shape[0], axis= 0)

        if col_exp != col_org:
            expanded_level = np.delete(expanded_level, lapl_pyr[0].shape[1], axis = 1)
        lapl_pyr[0] += expanded_level
        output = lapl_pyr[0]
        return output
    return output
    #...
    """
    return output

if __name__ == "__main__":    
    print("Practica 1 - Tarea 3 - Test autoevaluación\n")                
    print("Tests completados = " + str(test_p1_tarea3(precision=2))) 