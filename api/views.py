import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
import numpy as np
import os
import io
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings

def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_negative(image):
    return cv2.bitwise_not(image)

def apply_rgb_channels(image):
    b, g, r = cv2.split(image)
    return r

def apply_black_and_white(image, threshold_val=127):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw_image = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    return bw_image

def apply_threshold(image, threshold_val=127, max_val=255, threshold_type=cv2.THRESH_BINARY):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray, threshold_val, max_val, threshold_type)
    return thresholded_image

def apply_solarize(image):
    solarized_image = np.where(image < 128, image, 255 - image).astype(np.uint8)
    return solarized_image

def apply_posterize(image, levels=4):
    if levels < 2: levels = 2
    step = 255 / (levels - 1)
    posterized_image = (np.floor(image / step) * step).astype(np.uint8)
    return posterized_image

def apply_false_color(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    false_color_image = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return false_color_image

def apply_remove_color(image, color_to_remove='red'):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = None
    if color_to_remove == 'red':
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 100])
        upper2 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif color_to_remove == 'green':
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
    elif color_to_remove == 'blue':
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    else:
        return image
    if mask is not None:
        mask_inv = cv2.bitwise_not(mask)
        result = cv2.bitwise_and(image, image, mask=mask_inv)
        return result
    return image

def apply_increase_contrast(image, alpha=1.5, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def apply_increase_brightness(image, alpha=1.0, beta=50):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def generate_histogram(image, storage_path, filename):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    plt.figure(figsize=(8, 6))
    plt.title('Histograma de la Imagen')
    plt.xlabel('Intensidad de Píxel')
    plt.ylabel('Número de Píxeles')
    plt.plot(hist, color='black')
    plt.xlim([0, 256])
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()

    histogram_filename = f'histogram_{filename}'
    histogram_filepath = os.path.join(storage_path, histogram_filename)

    with open(histogram_filepath, 'wb') as f:
        f.write(buf.getvalue())
    return histogram_filename

def apply_equalize_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray)
    return equalized_image

def apply_gaussian_blur(image, ksize=(5, 5)):
    return cv2.GaussianBlur(image, ksize, 0)

def apply_average_blur(image, ksize=(5, 5)):
    return cv2.blur(image, ksize)

def apply_high_pass_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    high_pass_image = cv2.convertScaleAbs(laplacian)
    return high_pass_image

def apply_edge_detection_sobel(image, ksize=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return edges

def apply_edge_detection_canny(image, low_threshold=100, high_threshold=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contour_image

def apply_morphology(image, op_type='dilate', kernel_size=(5,5)):
    kernel = np.ones(kernel_size, np.uint8)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if op_type == 'erode':
        processed_img = cv2.erode(image, kernel, iterations=1)
    elif op_type == 'dilate':
        processed_img = cv2.dilate(image, kernel, iterations=1)
    elif op_type == 'open':
        processed_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif op_type == 'close':
        processed_img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    else:
        processed_img = image
    return processed_img

def resize_image(image, scale_percent=50):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image

def calculate_area_perimeter_centroid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_image = image.copy()
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(annotated_image, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(annotated_image, f'Centroide: ({cX}, {cY})', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_image, 'No se encontro centroide', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_image, f'Area: {area:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_image, f'Perimetro: {perimeter:.2f}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(annotated_image, 'No se encontraron contornos', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return annotated_image

def detect_face_and_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    if face_cascade.empty() or eye_cascade.empty():
        print("Advertencia: Archivos de cascada de Haar no encontrados. La detección de rostro/ojos no funcionará.")
        return image

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    detected_image = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(detected_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = detected_image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return detected_image



def index(request):
    context = {}
    fs = FileSystemStorage(location=settings.MEDIA_ROOT)

    if request.method == 'POST' and 'image' in request.FILES:
        image_file = request.FILES['image']
        original_filename = fs.save(image_file.name, image_file)

        request.session['uploaded_image_filename'] = original_filename
        request.session.save() # Ensure session is saved

        context['original_image_url'] = fs.url(original_filename)
        return render(request, 'index.html', context)

    elif request.method == 'POST' and 'filtro' in request.POST:
        filtro = request.POST.get('filtro')

        original_filename = request.session.get('uploaded_image_filename')

        if not original_filename or not fs.exists(original_filename):
            context['error'] = 'No se encontró la imagen original. Por favor, suba una imagen primero.'
            return render(request, 'index.html', context)

        uploaded_image_path = fs.path(original_filename)
        img = cv2.imread(uploaded_image_path)

        if img is None:
            context['error'] = 'Error: No se pudo cargar la imagen original desde el servidor. Archivo corrupto o formato no soportado.'
            if 'uploaded_image_filename' in request.session:
                del request.session['uploaded_image_filename']
            request.session.save()
            return render(request, 'index.html', context)

        processed_img = None
        histogram_url = None

        if filtro == 'grises':
            processed_img = apply_grayscale(img)
        elif filtro == 'negativo':
            processed_img = apply_negative(img)
        elif filtro == 'canales_rgb':
            processed_img = apply_rgb_channels(img)
        elif filtro == 'blanco_negro':
            processed_img = apply_black_and_white(img)
        elif filtro == 'umbral':
            processed_img = apply_threshold(img, threshold_val=100)
        elif filtro == 'solarizado':
            processed_img = apply_solarize(img)
        elif filtro == 'posterizado':
            processed_img = apply_posterize(img, levels=8)
        elif filtro == 'falso_color':
            processed_img = apply_false_color(img)
        elif filtro == 'eliminar_color_rojo':
            processed_img = apply_remove_color(img, color_to_remove='red')
        elif filtro == 'aumentar_contraste':
            processed_img = apply_increase_contrast(img)
        elif filtro == 'aumentar_brillo':
            processed_img = apply_increase_brightness(img)
        elif filtro == 'histograma':
            histogram_filename = generate_histogram(img, settings.MEDIA_ROOT, original_filename)
            histogram_url = fs.url(histogram_filename)
            processed_img = img
        elif filtro == 'ecualizar_histograma':
            processed_img = apply_equalize_histogram(img)
        elif filtro == 'filtro_gaussiano':
            processed_img = apply_gaussian_blur(img)
        elif filtro == 'filtro_promedio':
            processed_img = apply_average_blur(img)
        elif filtro == 'filtro_pasa_alto':
            processed_img = apply_high_pass_filter(img)
        elif filtro == 'deteccion_bordes_sobel':
            processed_img = apply_edge_detection_sobel(img)
        elif filtro == 'deteccion_bordes_canny':
            processed_img = apply_edge_detection_canny(img)
        elif filtro == 'contornos':
            processed_img = find_contours(img)
        elif filtro == 'morfologia_dilatacion':
            processed_img = apply_morphology(img, op_type='dilate')
        elif filtro == 'cambiar_tamano':
            processed_img = resize_image(img, scale_percent=50)
        elif filtro == 'area_perimetro_centroide':
            processed_img = calculate_area_perimeter_centroid(img)
        elif filtro == 'detectar_rostro_ojos':
            processed_img = detect_face_and_eyes(img)
        else:
            processed_img = img


        if processed_img is not None and filtro != 'histograma':

            if len(processed_img.shape) == 2:
                processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
            else:
                processed_img_bgr = processed_img

            base_name, ext = os.path.splitext(original_filename)
            processed_filename = f'{base_name}_processed_{filtro}{ext}'
            processed_image_path = fs.path(processed_filename)

            try:
                cv2.imwrite(processed_image_path, processed_img_bgr)
                context['processed_image_url'] = fs.url(processed_filename)
            except Exception as e:
                context['error'] = f'Error al guardar la imagen procesada: {e}'

        context['original_image_url'] = fs.url(original_filename)
        if histogram_url:
            context['histogram_image_url'] = histogram_url

        return render(request, 'index.html', context)

    else:
        original_filename = request.session.get('uploaded_image_filename')
        if original_filename and fs.exists(original_filename):
            context['original_image_url'] = fs.url(original_filename)
        return render(request, 'index.html', context)