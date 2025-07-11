import cv2
from p2_tarea1 import detectar_puntos_interes_harris
from p2_tarea2 import descripcion_puntos_interes
from p2_tarea3 import correspondencias_puntos_interes
from skimage.data import camera
import numpy as np  # Corrected import
#4.1
def four_one():
    image = camera()
    result3 = descripcion_puntos_interes(image,detectar_puntos_interes_harris(image),8,16,'hist')
    return result3[1]



#4.3 
def four_three():
    image_path = ['img\\EGaudi_1.jpg', 'img\\Mount_Rushmore1.jpg', 'img\\NotreDame1.jpg']
    image_path2 = ['img\\EGaudi_2.jpg', 'img\\Mount_Rushmore2.jpg', 'img\\NotreDame2.jpg']

    all_correspondences = []  # Initialize an empty list to store correspondences

    for i, image in enumerate(image_path):
        img1 = image_path[i]
        img2 = image_path2[i]

        gray_image1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
        gray_image2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

        x1 = detectar_puntos_interes_harris(gray_image1)
        x2 = detectar_puntos_interes_harris(gray_image2)

        nbins = 10
        y1, _ = descripcion_puntos_interes(gray_image1, x1, vtam=10, nbins=nbins)
        y2, _ = descripcion_puntos_interes(gray_image2, x2, vtam=10, nbins=nbins)

        y1 = y1.reshape((y1.size // nbins, nbins))
        y2 = y2.reshape((y2.size // nbins, nbins))

        z = correspondencias_puntos_interes(y1, y2, tipoCorr="mindist")
        
        if z.size > 0:  # Check if the array is not empty
            all_correspondences.append(z)


    if all_correspondences:  # Check if the list is not empty
        all_correspondences = np.vstack(all_correspondences)  # Convert the list to a numpy array


    return all_correspondences  # Return the accumulated correspondences

def main():
    print("4.1 camera correspondencias")
    x = four_one()
    print(x)
    
    print("images Correspondencias")
    y = four_three()
    print(y)

if __name__ == "__main__":
    main()