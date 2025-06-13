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
from django.urls import reverse


def apply_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def apply_negative(image):
    return cv2.bitwise_not(image)


def apply_rgb_channels(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    b, g, r = cv2.split(image)
    return r


def apply_black_and_white(image, level=50):
    threshold_val = int(255 * (level / 100))
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, bw_image = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    return bw_image


def apply_threshold(image, threshold_val=127, max_val=255, threshold_type=cv2.THRESH_BINARY):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, thresholded_image = cv2.threshold(gray, threshold_val, max_val, threshold_type)
    return thresholded_image


def apply_solarize(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    solarized_image = np.where(image < 128, image, 255 - image).astype(np.uint8)
    return solarized_image


def apply_posterize(image, levels=4):
    if levels < 2: levels = 2
    step = 255 / (levels - 1)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    posterized_image = (np.floor(image / step) * step).astype(np.uint8)
    return posterized_image


def apply_false_color(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    false_color_image = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return false_color_image


def apply_remove_color(image, color_to_remove='red'):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
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


def apply_increase_contrast(image, level=50):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    alpha = 1.0 + (level / 100) * 2.0
    beta = 0
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_increase_brightness(image, level=50):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    beta = int((level / 100) * 200 - 100)
    alpha = 1.0
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def generate_histogram(image, storage_path, filename):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
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
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    equalized_image = cv2.equalizeHist(gray)
    return equalized_image


def apply_gaussian_blur(image, ksize=(5, 5)):
    return cv2.GaussianBlur(image, ksize, 0)


def apply_average_blur(image, ksize=(5, 5)):
    return cv2.blur(image, ksize)


def apply_high_pass_filter(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    high_pass_image = cv2.convertScaleAbs(laplacian)
    return high_pass_image


def apply_edge_detection_sobel(image, ksize=3):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return edges


def apply_edge_detection_canny(image, low_threshold=100, high_threshold=200):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


def find_contours(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contour_image


def apply_morphology(image, op_type='dilate', kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    original_is_color = (len(image.shape) == 3)
    if original_is_color:
        image_to_process = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_to_process = image

    if op_type == 'erode':
        processed_img = cv2.erode(image_to_process, kernel, iterations=1)
    elif op_type == 'dilate':
        processed_img = cv2.dilate(image_to_process, kernel, iterations=1)
    elif op_type == 'open':
        processed_img = cv2.morphologyEx(image_to_process, cv2.MORPH_OPEN, kernel)
    elif op_type == 'close':
        processed_img = cv2.morphologyEx(image_to_process, cv2.MORPH_CLOSE, kernel)
    else:
        processed_img = image_to_process

    if original_is_color and len(processed_img.shape) == 2:
        return cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    return processed_img


def resize_image(image, scale_percent=50):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


def calculate_area_perimeter_centroid(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_image = image.copy()
    if len(annotated_image.shape) == 2:
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)

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
    if len(image.shape) == 2:
        detected_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        detected_image = image.copy()

    gray = cv2.cvtColor(detected_image, cv2.COLOR_BGR2GRAY)

    face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    eye_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    if face_cascade.empty() or eye_cascade.empty():
        print("Advertencia: Archivos de cascada de Haar no encontrados. La detección de rostro/ojos no funcionará.")
        return detected_image

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(detected_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = detected_image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return detected_image


def index(request):
    context = {}
    fs = FileSystemStorage(location=settings.MEDIA_ROOT)

    if request.method == 'POST' and 'image' in request.FILES:
        image_file = request.FILES['image']
        original_filename_base, original_filename_ext = os.path.splitext(image_file.name)
        unique_original_filename = f"{original_filename_base}_{os.urandom(8).hex()}{original_filename_ext}"
        original_filename = fs.save(unique_original_filename, image_file)

        request.session['uploaded_image_filename'] = original_filename
        request.session[
            'current_processed_image_filename'] = original_filename
        request.session.save()

        context['original_image_url'] = fs.url(original_filename)
        context['uploaded_image_filename'] = original_filename
        context['current_processed_image_filename'] = original_filename
        return render(request, 'index.html', context)

    elif request.method == 'POST':
        if 'reset_image' in request.POST:
            original_filename = request.session.get('uploaded_image_filename')
            if original_filename and fs.exists(original_filename):
                request.session['current_processed_image_filename'] = original_filename
                request.session.save()
                context['original_image_url'] = fs.url(original_filename)
                context['processed_image_url'] = fs.url(original_filename)
                context['uploaded_image_filename'] = original_filename
                context['current_processed_image_filename'] = original_filename
            else:
                context['error'] = 'No hay una imagen para reiniciar. Por favor, suba una imagen primero.'
            return render(request, 'index.html', context)


        current_processed_filename = request.session.get('current_processed_image_filename')
        original_filename = request.session.get('uploaded_image_filename')

        if not current_processed_filename or not fs.exists(current_processed_filename):
            context['error'] = 'No se encontró la imagen para aplicar el filtro. Por favor, suba una imagen primero.'
            if 'uploaded_image_filename' in request.session:
                del request.session['uploaded_image_filename']
            if 'current_processed_image_filename' in request.session:
                del request.session['current_processed_image_filename']
            request.session.save()
            return render(request, 'index.html', context)

        image_to_process_path = fs.path(current_processed_filename)
        img = cv2.imread(image_to_process_path)

        if img is None:
            context[
                'error'] = f'Error: No se pudo cargar la imagen actual ({current_processed_filename}). Archivo corrupto o formato no soportado.'
            if 'uploaded_image_filename' in request.session:
                del request.session['uploaded_image_filename']
            if 'current_processed_image_filename' in request.session:
                del request.session['current_processed_image_filename']
            request.session.save()
            return render(request, 'index.html', context)

        processed_img = None
        histogram_url = None
        current_filtro = None

        if 'filtro' in request.POST:
            current_filtro = request.POST['filtro']
        if current_filtro is None:
            if 'blanco_negro_level' in request.POST:
                current_filtro = 'blanco_negro'
            elif 'contraste_level' in request.POST:
                current_filtro = 'increase_contrast'
            elif 'brillo_level' in request.POST:
                current_filtro = 'increase_brightness'

        if current_filtro == 'grises':
            processed_img = apply_grayscale(img)
        elif current_filtro == 'negativo':
            processed_img = apply_negative(img)
        elif current_filtro == 'canales_rgb':
            processed_img = apply_rgb_channels(img)
        elif current_filtro == 'blanco_negro':
            level = int(request.POST.get('blanco_negro_level', 50))
            processed_img = apply_black_and_white(img, level=level)
        elif current_filtro == 'increase_contrast':
            level = int(request.POST.get('contraste_level', 50))
            processed_img = apply_increase_contrast(img, level=level)
        elif current_filtro == 'increase_brightness':
            level = int(request.POST.get('brillo_level', 50))
            processed_img = apply_increase_brightness(img, level=level)
        elif current_filtro == 'umbral':
            processed_img = apply_threshold(img, threshold_val=100)
        elif current_filtro == 'solarizado':
            processed_img = apply_solarize(img)
        elif current_filtro == 'posterizado':
            processed_img = apply_posterize(img, levels=4)
        elif current_filtro == 'falso_color':
            processed_img = apply_false_color(img)
        elif current_filtro == 'eliminar_color_rojo':
            processed_img = apply_remove_color(img, color_to_remove='red')
        elif current_filtro == 'histograma':
            histogram_filename = generate_histogram(img, settings.MEDIA_ROOT, current_processed_filename)
            histogram_url = fs.url(histogram_filename)
            processed_img = img
        elif current_filtro == 'ecualizar_histograma':
            processed_img = apply_equalize_histogram(img)
        elif current_filtro == 'filtro_gaussiano':
            processed_img = apply_gaussian_blur(img)
        elif current_filtro == 'filtro_promedio':
            processed_img = apply_average_blur(img)
        elif current_filtro == 'filtro_pasa_alto':
            processed_img = apply_high_pass_filter(img)
        elif current_filtro == 'deteccion_bordes_sobel':
            processed_img = apply_edge_detection_sobel(img)
        elif current_filtro == 'deteccion_bordes_canny':
            processed_img = apply_edge_detection_canny(img)
        elif current_filtro == 'contornos':
            processed_img = find_contours(img)
        elif current_filtro == 'morfologia_dilatacion':
            processed_img = apply_morphology(img, op_type='dilate')
        elif current_filtro == 'cambiar_tamano':
            processed_img = resize_image(img, scale_percent=50)
        elif current_filtro == 'area_perimetro_centroide':
            processed_img = calculate_area_perimeter_centroid(img)
        elif current_filtro == 'detectar_rostro_ojos':
            processed_img = detect_face_and_eyes(img)
        else:
            processed_img = img

        if processed_img is not None and current_filtro != 'histograma':
            if len(processed_img.shape) == 2:
                processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
            else:
                processed_img_bgr = processed_img

            base_name, ext = os.path.splitext(current_processed_filename)
            if len(base_name) > 50:
                base_name = base_name[-50:]

            safe_current_filtro = "".join(c for c in current_filtro if c.isalnum() or c in ['_', '-']).lower()

            unique_hash = os.urandom(4).hex()

            if current_filtro:
                if current_filtro == 'blanco_negro':
                    level = request.POST.get('blanco_negro_level', '50')
                    new_processed_filename = f'{base_name}_{safe_current_filtro}_{level}_{unique_hash}{ext}'
                elif current_filtro == 'increase_contrast':
                    level = request.POST.get('contraste_level', '50')
                    new_processed_filename = f'{base_name}_{safe_current_filtro}_{level}_{unique_hash}{ext}'
                elif current_filtro == 'increase_brightness':
                    level = request.POST.get('brillo_level', '50')
                    new_processed_filename = f'{base_name}_{safe_current_filtro}_{level}_{unique_hash}{ext}'
                else:
                    new_processed_filename = f'{base_name}_{safe_current_filtro}_{unique_hash}{ext}'
            else:
                new_processed_filename = f'{base_name}_processed_{unique_hash}{ext}'

            processed_image_path = fs.path(new_processed_filename)

            try:
                cv2.imwrite(processed_image_path, processed_img_bgr)
                context['processed_image_url'] = fs.url(new_processed_filename)
                request.session['current_processed_image_filename'] = new_processed_filename
                request.session.save()
            except Exception as e:
                context['error'] = f'Error al guardar la imagen procesada: {e}'

        if original_filename and fs.exists(original_filename):
            context['original_image_url'] = fs.url(original_filename)
            context['uploaded_image_filename'] = original_filename
        else:
            if 'uploaded_image_filename' in request.session:
                del request.session['uploaded_image_filename']
            if 'current_processed_image_filename' in request.session:
                del request.session['current_processed_image_filename']
            request.session.save()
            context['error'] = 'La imagen original no está disponible. Por favor, suba una imagen nueva.'

        if 'processed_image_url' not in context and current_processed_filename and fs.exists(
                current_processed_filename):
            context['processed_image_url'] = fs.url(current_processed_filename)
        elif 'processed_image_url' not in context and original_filename and fs.exists(original_filename):
            context['processed_image_url'] = fs.url(original_filename)

        context['current_processed_image_filename'] = request.session.get('current_processed_image_filename', '')

        if histogram_url:
            context['histogram_image_url'] = histogram_url

        return render(request, 'index.html', context)

    else:
        original_filename = request.session.get('uploaded_image_filename')
        current_processed_filename = request.session.get('current_processed_image_filename')

        if original_filename and fs.exists(original_filename):
            context['original_image_url'] = fs.url(original_filename)
            context['uploaded_image_filename'] = original_filename

            if current_processed_filename and fs.exists(current_processed_filename):
                context['processed_image_url'] = fs.url(current_processed_filename)
                context['current_processed_image_filename'] = current_processed_filename
            else:
                context['processed_image_url'] = fs.url(original_filename)
                context['current_processed_image_filename'] = original_filename

        return render(request, 'index.html', context)