import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from skimage.color import deltaE_ciede2000, rgb2lab

app = Flask(__name__)
CORS(app)

# --- Calibration curve fitted from real test-card measurements ---
# dose = a*delta_e^2 + b*delta_e + c
A, B, C = 0.00354432, 0.8518152, 0.35734949

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Server is running!"}), 200


@app.route('/analyze', methods=['POST'])
def analyze_dosimeter():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['file']
    np_img = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return jsonify({"status": "Error", "message": "Marker not found. Make sure the wristband is fully visible and well-lit."}), 400

    dst_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    src_pts = corners[0][0].astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (300, 300))

    white_swatch = warped[20:60, 220:260]
    mean_bgr = cv2.mean(white_swatch)[:3]
    target_rgb = np.array([240.0, 240.0, 240.0])
    measured_rgb = np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float32)
    gain = target_rgb / np.maximum(measured_rgb, 1.0)

    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized_rgb = np.clip(warped_rgb * gain, 0, 255).astype(np.uint8)

    unexposed_roi = normalized_rgb[120:160, 20:60]
    avg_unexposed_rgb = np.mean(unexposed_roi, axis=(0, 1)).reshape(1, 1, 3) / 255.0
    unexposed_lab = rgb2lab(avg_unexposed_rgb)

    patch_roi = normalized_rgb[140:240, 140:240]
    avg_patch_rgb = np.mean(patch_roi, axis=(0, 1)).reshape(1, 1, 3) / 255.0
    current_lab = rgb2lab(avg_patch_rgb)

    delta_e = float(deltaE_ciede2000(unexposed_lab, current_lab)[0][0])

    dose_ppm_hours = round(A * delta_e**2 + B * delta_e + C, 2)
    SAFE_LIMIT = 10.0

    safety_label = "SAFE" if dose_ppm_hours <= SAFE_LIMIT else "DANGER"

    return jsonify({
        "delta_e": round(delta_e, 2),
        "ppm_hours": dose_ppm_hours,
        "status": safety_label,
        "safe_limit_ppm_hours": SAFE_LIMIT,
        "message": (
            f"Exposure is within safe limits ({dose_ppm_hours} / {SAFE_LIMIT} ppm·h)"
            if safety_label == "SAFE"
            else f"⚠ Exposure exceeds safe limit! ({dose_ppm_hours} / {SAFE_LIMIT} ppm·h)"
        )
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
