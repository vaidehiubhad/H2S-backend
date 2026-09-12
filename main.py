import os
import cv2
import numpy as np
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
from skimage.color import deltaE_ciede2000, rgb2lab

app = Flask(__name__)
CORS(app)

A, B, C = 0.00354432, 0.8518152, 0.35734949
SAFE_LIMIT = 8.0
STANDARD_REFERENCE = "ACGIH TLV 8-hr TWA (1 ppm) | OSHA PEL ceiling 20 ppm | NIOSH REL ceiling 10 ppm/10min"

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

    if ids is None or len(ids) < 3:
        return jsonify({
            "status": "Error",
            "message": "Could not find enough corner markers. Ensure at least 3 of the 4 corner markers are visible and unobstructed."
        }), 400

    ids = ids.flatten()
    corner_pick = {0: 0, 1: 1, 2: 2, 3: 3}
    CARD_SIZE = 300
    ideal_positions = {0: [0, 0], 1: [CARD_SIZE, 0], 2: [CARD_SIZE, CARD_SIZE], 3: [0, CARD_SIZE]}

    src_pts, dst_pts = [], []
    for i, marker_id in enumerate(ids):
        if marker_id in corner_pick:
            src_pts.append(corners[i][0][corner_pick[marker_id]])
            dst_pts.append(ideal_positions[marker_id])

    if len(src_pts) < 3:
        return jsonify({"status": "Error", "message": "Not enough visible corner markers to align the wristband."}), 400

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    matrix, _ = cv2.findHomography(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (CARD_SIZE, CARD_SIZE))

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
    safety_label = "SAFE" if dose_ppm_hours <= SAFE_LIMIT else "DANGER"

    return jsonify({
        "delta_e": round(delta_e, 2),
        "ppm_hours": dose_ppm_hours,
        "status": safety_label,
        "safe_limit_ppm_hours": SAFE_LIMIT,
        "standard_reference": STANDARD_REFERENCE,
        "markers_detected": len(src_pts),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": (
            f"✅ Exposure within safe limits ({dose_ppm_hours} / {SAFE_LIMIT} ppm·h, per ACGIH TWA)"
            if safety_label == "SAFE"
            else f"⚠ DANGER: Exposure exceeds safe limit! ({dose_ppm_hours} / {SAFE_LIMIT} ppm·h) — remove worker from area and seek fresh air immediately."
        ),
        "disclaimer": "Prototype device for demonstration only. Not certified for occupational safety compliance. Always follow official workplace gas monitoring protocols."
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
