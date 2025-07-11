# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Tarea 3:  Similitud y correspondencia de puntos de interes

# AUTOR1: APELLIDO1 APELLIDO1, NOMBRE1
# AUTOR2: APELLIDO2 APELLIDO2, NOMBRE2
# PAREJA/TURNO: NUMERO_PAREJA/NUMERO_TURNO

# librerias y paquetes por defecto
import numpy as np
from p2_tests import test_p2_tarea3

# Incluya aqui las librerias que necesite en su codigo
# ...

def correspondencias_puntos_interes(descriptores_imagen1, descriptores_imagen2, tipoCorr='mindist',max_distancia=25):
    """
    # Esta funcion determina la correspondencias entre dos conjuntos de descriptores mediante
    # el calculo de la similitud entre los descriptores.
    #
    # El parametro 'tipoCorr' determina el criterio de similitud aplicado 
    # para establecer correspondencias entre pares de descriptores:
    #   - Criterio 'mindist': minima distancia euclidea entre descriptores 
    #                         menor que el umbral 'max_distancia'
    #  
    # Argumentos de entrada:
    #   - descriptores1: numpy array con dimensiones [numero_descriptores, longitud_descriptor] 
    #                    con los descriptores de los puntos de interes de la imagen 1.        
    #   - descriptores2: numpy array con dimensiones [numero_descriptores, longitud_descriptor] 
    #                    con los descriptores de los puntos de interes de la imagen 2.        
    #   - tipoCorr: cadena de caracteres que indica el tipo de criterio para establecer correspondencias
    #   - max_distancia: valor de tipo double o float utilizado por el criterio 'mindist' y 'nndr', 
    #                    que determina si se aceptan correspondencias entre descriptores 
    #                    con distancia minima menor que 'max_distancia' 
    #
    # Argumentos de salida
    #   - correspondencias: numpy array con dimensiones [numero_correspondencias, 2] de tipo int64 
    #                       que determina correspondencias entre descriptores de imagen 1 e imagen 2.
    #                       Por ejemplo: 
    #                       correspondencias[0,:]=[5,22] significa que el descriptor 5 de la imagen 1 
    #                                                  corresponde con el descriptor 22 de la imagen 2. 
    #                       correspondencias[1,:]=[6,23] significa que el descriptor 6 de la imagen 1 
    #                                                  corresponde con el descriptor 23 de la imagen 2.
    #
    # NOTA: no modificar los valores por defecto de las variables de entrada tipoCorr y max_distancia, 
    #       pues se utilizan para verificar el correcto funciomaniento de esta funcion
    #
    # CONSIDERACIONES: 
    # 1) La funcion no debe permitir correspondencias de uno a varios descriptores. Es decir, 
    #   un descriptor de la imagen 1 no puede asignarse a multiples descriptores de la imagen 2 
    # 2) En el caso de que existan varios descriptores de la imagen 2 con la misma distancia minima 
    #    con algún descriptor de la imagen 1, seleccione el descriptor de la imagen 2 con 
    #    indice/posicion menor. Por ejemplo, si las correspondencias [5,22] y [5,23] tienen la misma
    #    distancia minima, seleccione [5,22] al ser el indice 22 menor que 23
    """    
    
    
    if tipoCorr == "nndr":
        corr_list = []
        taken = set()

        for i,descImg1 in enumerate(descriptores_imagen1):
            mindist = None
            minj = None

            for j,descImg2 in enumerate(descriptores_imagen2):
                if j in taken:
                    continue

                dist = np.linalg.norm(descImg1 - descImg2)              
                if mindist is None or dist < mindist:
                    mindist = dist
                    minj = j

            if mindist is not None and mindist <= max_distancia:
                corr_list.append([i,minj]) 
                taken.add(minj)

        correspondencias = np.array(corr_list, dtype=np.int64)
        return correspondencias
    
    #4.4
    elif tipoCorr == "mindist":

        corr_list = []
        taken = set()

        for i,descImg1 in enumerate(descriptores_imagen1):
            mindist = None
            minj = None
            secondmin = None
            secondminj = None

            for j,descImg2 in enumerate(descriptores_imagen2):
                if j in taken:
                    continue

                dist = np.linalg.norm(descImg1 - descImg2)              
                if mindist is None or dist < mindist:
                    mindist = dist
                    minj = j

                elif secondmin is None or dist < secondmin:
                    secondmin = dist
                    secondminj = j

            if secondmin is None or secondmin == 0:
                NNDR = 0
            else:
                NNDR = mindist / secondmin

            if mindist is not None and mindist <= max_distancia and NNDR < 0.25:
                corr_list.append([i,minj]) 
                taken.add(minj)

        correspondencias = np.array(corr_list, dtype=np.int64)
        return correspondencias

if __name__ == "__main__":
    print("Practica 2 - Tarea 3 - Test autoevaluación\n")                

    ## tests correspondencias tipo 'minDist' (tarea 3a)
    #print("Tests completados = " + str(test_p2_tarea3(disptime=2,stop_at_error=True,debug=True,tipoDesc='hist',tipoCorr='mindist'))) #analizar todas las imagenes con descriptor 'hist' y ver errores
    #print("Tests completados = " + str(test_p2_tarea3(disptime=2,stop_at_error=False,debug=False,tipoDesc='hist',tipoCorr='mindist'))) #analizar todas las imagenes con descriptor 'hist'
    print("Tests completados = " + str(test_p2_tarea3(disptime=1,stop_at_error=False,debug=False,tipoDesc='mag-ori',tipoCorr='mindist'))) #analizar todas las imagenes con descriptor 'mag-ori'

    

    #how good does hist and magori work?