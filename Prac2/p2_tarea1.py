# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Tarea 1: Deteccion de puntos de interes con Harris corner detector.

# AUTOR1: APELLIDO1 APELLIDO1, NOMBRE1
# AUTOR2: APELLIDO2 APELLIDO2, NOMBRE2
# PAREJA/TURNO: NUMERO_PAREJA/NUMERO_TURNO

# librerias y paquetes por defecto
import numpy as np
from p2_tests import test_p2_tarea1
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from skimage.feature import corner_peaks
# Incluya aqui las librerias que necesite en su codigo
# ...

def detectar_puntos_interes_harris(imagen, sigma=1, k=0.05, threshold_rel=0.2):
    # Convierte la imagen al formato float en el rango [0, 1]
    imagen = imagen.astype(np.float) / 255.0

    # Calcula las derivadas utilizando filtros de Sobel
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Ix = convolve2d(imagen, kernel_x, mode='same')
    Iy = convolve2d(imagen, kernel_y, mode='same')

    # Calcula los elementos de la matriz M
    Ix2 = gaussian_filter(Ix**2, sigma, mode='constant')
    Iy2 = gaussian_filter(Iy**2, sigma, mode='constant')
    Ixy = gaussian_filter(Ix * Iy, sigma, mode='constant')

    # Calcula la respuesta de Harris
    det_M = Ix2 * Iy2 - Ixy**2
    trace_M = Ix2 + Iy2
    harris_response = det_M - k * (trace_M**2)

    # Aplica umbral y encuentra máximos locales
    threshold = threshold_rel * np.max(harris_response)
    
    coords_esquinas = corner_peaks(harris_response, min_distance=5, threshold_abs=threshold, threshold_rel=0)

    return coords_esquinas

if __name__ == "__main__":    
    print("Practica 2 - Tarea 1 - Test autoevaluación\n")                
    
    print("Tests completados = " + str(test_p2_tarea1(disptime=-1,stop_at_error=False,debug=False))) #analizar todos los casos sin pararse en errores
    #print("Tests completados = " + str(test_p2_tarea1(disptime=1,stop_at_error=False,debug=False))) #analizar y visualizar todos los casos sin pararse en errores
    #print("Tests completados = " + str(test_p2_tarea1(disptime=-1,stop_at_error=True,debug=False))) #analizar todos los casos y pararse en errores 
    #print("Tests completados = " + str(test_p2_tarea1(disptime=-1,stop_at_error=True,debug=True))) #analizar todos los casos, pararse en errores y mostrar informacion

    