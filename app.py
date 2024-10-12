import gradio as gr
from PIL import Image, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Fonctions de traitement d'image
def load_image(image):
    return image

def apply_negative(image):
    img_np = np.array(image)
    negative = 255 - img_np
    return Image.fromarray(negative)

def binarize_image(image, threshold):
    img_np = np.array(image.convert('L'))
    _, binary = cv2.threshold(img_np, threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary)

def resize_image(image, width, height):
    return image.resize((width, height))

def rotate_image(image, angle):
    return image.rotate(angle)

def histo_gray(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title('Histogramme des niveaux de gris')
    plt.xlabel('Intensité des pixels')
    plt.ylabel('Nombre de pixels')
    plt.show()
    return hist

def filtre_gauss(image, kernel_width, kernel_heigth):
    return cv2.GaussianBlur(image, (kernel_width, kernel_heigth), 0)

def erosion(image, taille):
    return image.filter(ImageFilter.MinFilter(taille_e))

def dilatation(image, taille):
    return image.filter(ImageFilter.MaxFilter(taille_d))

def extract_edges(image):
    image_sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 5)
    image_sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 5)
    return image_sobelx, image_sobely

# Interface Gradio
def image_processing(image, operation, threshold=128, width=100, height=100, angle=0):
    if operation == "Négatif":
        return apply_negative(image)
    elif operation == "Binarisation":
        return binarize_image(image, threshold)
    elif operation == "Redimensionner":
        return resize_image(image, width, height)
    elif operation == "Rotation":
        return rotate_image(image, angle)
    elif operation == "Histogramme des niveaux de gris":
        return histo_gray(image)
    elif operation == "Filtre gaussien":
        return filtre_gauss(image, kernel_width, kernel_heigth)
    elif operation == "Erosion":
        return erosion(image, taille_e)
    elif operation == "Dilatation":
        return dilatation(image, taille_d)
    elif operation == "Extraction de contours":
        return extract_edges(image)       

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Projet de Traitement d'Image")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Charger Image")
        operation = gr.Radio(["Négatif", "Binarisation", "Redimension", "Rotation", "Histogramme des niveaux de gris", "Filtre gaussien", "Extraction de contours", "Erosion", "Dilatation"], label="Opération")
        threshold = gr.Slider(0, 255, 128, label="Seuil de binarisation", visible=False)
        width = gr.Number(value=100, label="Largeur", visible=False)
        height = gr.Number(value=100, label="Hauteur", visible=False)
        angle = gr.Number(value=0, label="Angle de Rotation", visible=False)
        kernel_width = gr.Number(value=5, label="Largeur du kernel du filtre gaussien", visible=False)
        kernel_heigth = gr.Number(value=5, label="Hauteur du kernel du filtre gaussien", visible=False)
        taille_e = gr.Number(value=3, label="Taille du filtre pour l'érosion", visible=False)
        taille_d = gr.Number(value=3, label="Taille du filtre pour la dilatation", visible=False)

    image_output = gr.Image(label="Image Modifiée")

    submit_button = gr.Button("Appliquer")
    submit_button.click(image_processing, inputs=[image_input, operation, threshold, width, height, angle], outputs=image_output)

# Lancer l'application Gradio
demo.launch()
