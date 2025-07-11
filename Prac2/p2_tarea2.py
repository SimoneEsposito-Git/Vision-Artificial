# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Tarea 2: Descripcion de puntos de interes mediante histogramas.

# AUTOR1: APELLIDO1 APELLIDO1, NOMBRE1
# AUTOR2: APELLIDO2 APELLIDO2, NOMBRE2
# PAREJA/TURNO: NUMERO_PAREJA/NUMERO_TURNO

# librerias y paquetes por defecto
import numpy as np

from skimage.data import camera
from scipy.ndimage import sobel
from numpy import rad2deg, arctan2, linspace, digitize
from p2_tests import test_p2_tarea2
from p2_tarea1 import detectar_puntos_interes_harris

# Incluya aqui las librerias que necesite en su codigo
# ...

def descripcion_puntos_interes(imagen, coords_esquinas, vtam = 8, nbins = 16, tipoDesc='hist'):
    """
    # Esta funcion describe puntos de interes de una imagen mediante histogramas, analizando 
    # vecindarios con dimensiones "vtam+1"x"vtam+1" centrados en las coordenadas de cada punto de interes
    #   
    # La descripcion obtenida depende del parametro 'tipoDesc'
    #   - Caso 'hist': histograma normalizado de valores de gris 
    #   - Caso 'mag-ori': histograma de orientaciones de gradiente
    #
    # En el caso de que existan puntos de interes en los bordes de la imagen, el descriptor no
    # se calcula y el punto de interes se elimina de la lista <new_coords_esquinas> que devuelve
    # esta funcion. Esta lista indica los puntos de interes para los cuales existe descriptor.
    #
    # Argumentos de entrada:
    #   - imagen: numpy array con dimensiones [imagen_height, imagen_width].        
    #   - coords_esquinas: numpy array con dimensiones [num_puntos_interes, 2] con las coordenadas 
    #                      de los puntos de interes detectados en la imagen. Tipo int64
    #                      Cada punto de interes se encuentra en el formato [fila, columna]
    #   - vtam: valor de tipo entero que indica el tamaño del vecindario a considerar para
    #           calcular el descriptor correspondiente.
    #   - nbins: valor de tipo entero que indica el numero de niveles que tiene el histograma 
    #           para calcular el descriptor correspondiente.
    #   - tipoDesc: cadena de caracteres que indica el tipo de descriptor calculado
    #
    # Argumentos de salida
    #   - descriptores: numpy array con dimensiones [num_puntos_interes, nbins] con los descriptores 
    #                   de cada punto de interes (i.e. histograma de niveles de gris)
    #   - new_coords_esquinas: numpy array con dimensiones [num_puntos_interes, 2], solamente con las coordenadas 
    #                      de los puntos de interes descritos. Tipo int64  <class 'numpy.ndarray'>
    #
    # NOTA: no modificar los valores por defecto de las variables de entrada vtam y nbins, 
    #       pues se utilizan para verificar el correcto funciomaniento de esta funcion
    """
    # Iniciamos variables de salida
    descriptores = np.empty(shape=[0,0]) # iniciamos la variable de salida (numpy array)
    new_coords_esquinas = np.empty(shape=[0,0]) # iniciamos la variable de salida (numpy array)

    #incluya su codigo aqui
    #...    
    
    for esquina in coords_esquinas:
        row, col = esquina
        offset = int((vtam+1)/2)
        # Asegura que el vecindario esté completamente contenido en la imagen
        if (row-offset) < 0 or (row + (vtam - offset)+1)>imagen.shape[0] or  (col-offset) < 0 or (col + (vtam - offset)+1)>imagen.shape[1]:
            break
            
        start_row, end_row = (row - offset), (row + (vtam - offset)+1)
        start_col, end_col = (col - offset), (col + (vtam - offset)+1)

        vecindario = imagen[start_row:end_row, start_col:end_col]
        
        if tipoDesc == 'hist':
            # Calcula el histograma normalizado
            hist, bin_edges = np.histogram(vecindario.flatten(), bins=nbins, range =(0,1))
            histsum = np.sum(hist)

            if histsum > 0:
                hist = hist / np.sum(hist)           
            
        
        elif tipoDesc == 'mag-ori':

            bin_edges = np.linspace(0, 360, nbins + 1)
            # Calcular el gradiente utilizando el operador Sobel
            grad_x = sobel(vecindario, axis=0, mode="reflect")
            grad_y = sobel(vecindario, axis=1, mode="reflect")

            # Calcular la magnitud y la orientación del gradiente
            magnitud = np.sqrt(grad_x**2 + grad_y**2)
            orientacion = rad2deg(arctan2(grad_x, grad_y)) 
            
            orientacion[orientacion < 0] += 360 

            bin_indices = np.digitize(orientacion, bin_edges) - 1

            hist, _ = np.histogram(bin_indices, bins=range(nbins + 1), weights=magnitud)

        
        descriptores = np.append(descriptores, hist)
        new_coords_esquinas = np.vstack([new_coords_esquinas, [row, col]]) if new_coords_esquinas.size else np.array([row, col])

    if len(descriptores) > 0:
        descriptores = descriptores.reshape(1,len(descriptores))
    return descriptores, new_coords_esquinas

if __name__ == "__main__":    
    print("Practica 2 - Tarea 2 - Test autoevaluación\n")                

    ## tests descriptor tipo 'hist' (tarea 2a)
    
    print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=False,debug=False,tipoDesc='hist'))) #analizar todas las imagenes y esquinas del test
    #print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=False,debug=False,tipoDesc='hist'))) #analizar todas las imagenes y esquinas del test, mostrar imagenes con resultados (1 segundo)
    #print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=True,debug=True,tipoDesc='hist'))) #analizar todas las imagenes y esquinas del test, pararse en errores y mostrar datos
    #print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=True,debug=True,tipoDesc='hist',imgIdx = 3, poiIdx = 7))) #analizar solamente imagen #2 y esquina #7    

    ## tests descriptor tipo 'mag-ori' (tarea 2b)
    #print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=False,debug=True,tipoDesc='mag-ori'))) #analizar todas las imagenes y esquinas del test
    #print("Tests completados = " + str(test_p2_tarea2(disptime=0.1,stop_at_error=False,debug=False,tipoDesc='mag-ori'))) #analizar todas las imagenes y esquinas del test, mostrar imagenes con resultados (1 segundo)
    #print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=True,debug=True,tipoDesc='mag-ori'))) #analizar todas las imagenes y esquinas del test, pararse en errores y mostrar datos
    #print("Tests completados = " + str(test_p2_tarea2(disptime=1,stop_at_error=True,debug=True,tipoDesc='mag-ori',imgIdx = 3,poiIdx = 7))) #analizar solamente imagen #1 y esquina #7       

    


