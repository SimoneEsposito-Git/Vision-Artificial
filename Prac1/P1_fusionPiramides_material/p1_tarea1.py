# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 1: Fusion de imagenes mediante piramides
# Tarea 1: metodos reduce y expand

# AUTOR1: ESPOSITO, SIMONE
# AUTOR2: BOBENKO, KONSTANTIN
# PAREJA/TURNO: 21/VIERNES
import numpy as np
import scipy.signal
import math 

from p1_tests import test_p1_tarea1
from p1_utils import generar_kernel_suavizado

def reduce(imagen):
    """  
    # Esta funcion implementa la operacion "reduce" sobre una imagen
    # 
    # Argumentos de entrada:
    #    imagen: numpy array de tamaño [imagen_height, imagen_width].
    # 
    # Devuelve:
    #    output: numpy array de tamaño [imagen_height/2, imagen_width/2] (output).
    #
        # NOTA: si imagen_height/2 o imagen_width/2 no son numeros enteros, 
        #        entonces se redondea al entero mas cercano por arriba 
        #        Por ejemplo, si la imagen es 5x7, la salida sera 3x4  
    """   
    output = np.empty(shape=[0,0]) # iniciamos la variable de salida (numpy array)
    kernel = generar_kernel_suavizado(0.4)
    convolucion = scipy.signal.convolve2d(imagen, kernel, 'same')

    reduced_height = math.ceil(imagen.shape[0] / 2)
    reduced_width = math.ceil(imagen.shape[1] / 2)
    output = convolucion[::2, ::2][:reduced_height, :reduced_width]

    #...
   
    return output  

def expand(imagen):
    """  
    # Esta funcion implementa la operacion "expand" sobre una imagen
    # 
    # Argumentos de entrada:
    #    imagen: numpy array de tamaño [imagen_height, imagen_width].
    #     
    # Devuelve:
    #    output: numpy array de tamaño [imagen_height*2, imagen_width*2].
    """ 
    output = np.empty(shape=[0,0]) # iniciamos la variable de salida (numpy array)
    imagen_expandida = np.zeros((imagen.shape[0]*2, imagen.shape[1]*2))
    imagen_expandida[::2, ::2] = imagen
    kernel = generar_kernel_suavizado(0.4)
    convolucion = scipy.signal.convolve2d(imagen_expandida, kernel, 'same')
    output = convolucion * 4

    return output

if __name__ == "__main__":    
    print("Practica 1 - Tarea 2 - Test autoevaluación\n")                
    print("Tests completados = " + str(test_p1_tarea1()))
