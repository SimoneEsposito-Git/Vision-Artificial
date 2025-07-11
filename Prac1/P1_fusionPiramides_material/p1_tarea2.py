# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 1: Fusion de imagenes mediante piramides
# Tarea 2: piramide Gaussiana y piramide laplaciana

# AUTOR1: APELLIDO1 APELLIDO1, NOMBRE1
# AUTOR2: APELLIDO2 APELLIDO2, NOMBRE2
# PAREJA/TURNO: NUMERO_PAREJA/NUMERO_TURNO
import numpy as np

from p1_tests import test_p1_tarea2
from p1_tarea1 import reduce, expand

def gaus_piramide(imagen, niveles): 
    """ 
    # Esta funcion crea una piramide Gaussiana a partir de una imagen
    #
    # Argumentos de entrada:
    #   imagen: numpy array de tamaño [imagen_height, imagen_width].
    #   niveles: entero positivo que especifica el numero de veces que se aplica la operacion 'reduce'.
    #           Si niveles=0 entonces se debe devolver una lista con la imagen de entrada
    #           Si niveles=1 entonces se debe realizar una operacion de reduccion               
    #
    # Devuelve:
    #   gauss_pyr: lista de numpy arrays con variable tamaño con "niveles+1" elementos.
    #       output[0] es la imagen de entrada
    #       output[i] es el nivel i de la piramide
    #  
    """ 
    gaus_pyr = []  # iniciamos la variable de salida (lista)
    for nivel in range(niveles+1):

        if nivel == 0:
            gaus_pyr.append(imagen)
        else:
            gaus_pyr.append(reduce(gaus_pyr[nivel-1]))

    #...
    for i in range(len(gaus_pyr)):
        print("level ", i, np.shape(gaus_pyr[i]))
    return gaus_pyr

def lapl_piramide(gaus_pyr):
    """ 
    # Esta funcion crea una piramide Laplaciana a partir de una piramide Gaussiana    
    #  
    # Argumentos de entrada:
    #   gauss_pyr: lista de numpy arrays creada con la funcion 'gauss_piramide'.               
    #
    # Devuelve:
    #   lapla_pyr: lista de numpy arrays con variable tamaño con "niveles+1" elementos.    
    #       lapla_pyr[0] es el primera nivel de la piramide que contiene bordes
    #       lapla_pyr[i] es el nivel i de la piramide que contiene bordes
    #       lapla_pyr[niveles-1] es una imagen (escala de grises)    
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
    
    lapl_pyr = []  # Initialize the output list

    for i in range(len(gaus_pyr) - 1):
        expanded_level = expand(gaus_pyr[i + 1])

        rows_exp, col_exp = expanded_level.shape
        rows_org, col_org = gaus_pyr[i].shape
        
        if rows_exp != rows_org:
            expanded_level = np.delete(expanded_level, gaus_pyr[i].shape[0], axis= 0)
            
        if col_exp != col_org:
            expanded_level = np.delete(expanded_level, gaus_pyr[i].shape[1], axis = 1)
        

        lapl_pyr.append(gaus_pyr[i] - expanded_level)

    lapl_pyr.append(gaus_pyr[-1])

    return lapl_pyr
   
if __name__ == "__main__":    
    print("Practica 1 - Tarea 2 - Test autoevaluación\n")                
    print("Tests completados = " + str(test_p1_tarea2()))