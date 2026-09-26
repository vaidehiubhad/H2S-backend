import os
import cv2
import numpy as np
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
from skimage.color import deltaE_ciede2000, rgb2lab

app = Flask(__name__)
CORS(app)

# Calibration anchored to US Patent 5,364,593 (lead acetate exposimeter):
# white-to-black color range spans approx. 5 to 300 ppm-hours
CAL_K, CAL_P = 1.36020, 1.18496
NOISE_FLOOR_DELTA_E = 1.2

SAFE_LIMIT = 8.0
DANGER_LIMIT = 20.0
STANDARD_REFERENCE = "ACGIH TLV 8-hr TWA (1 ppm) | OSHA PEL ceiling 20 ppm | NIOSH REL ceiling 10 ppm/10min"
CALIBRATION_REFERENCE = "Dose range anchored to US Patent 5,364,593 (lead acetate exposimeter, 5-300 ppm-hours white-to-black)"

# --- NEW: expiry/shelf-life patch ---
# EXPIRY_REFERENCE_RGB is the color of a FRESH, unexpired patch measured under
# standard lighting. PLACEHOLDER VALUE — replace once real strips are measured.
EXPIRY_REFERENCE_RGB = np.array([[[70.0, 130.0, 180.0]]])  # steel-blue, fresh/valid state
EXPIRY_DELTA_E_THRESHOLD = 5.0  # PLACEHOLDER — tune with accelerated-aging test data

# --- NEW: sealed reference cell (temperature/humidity compensation) ---
# SEALED_CELL_EXPECTED_RGB is the known, fixed printed-ink color of the sealed
# swatch, measured once under standard lab conditions. It does not react to H2S.
# PLACEHOLDER VALUE — replace once real sealed cells are measured.
SEALED_CELL_EXPECTED_RGB = np.array([[[140.0, 150.0, 140.0]]])
ENV_COMPENSATION_FACTOR = 1.0  # PLACEHOLDER — tune with real temp/humidity trials

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    out = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, 1), 1 / 2.4) - 0.055)
    return np.clip(out * 255.0, 0, 255)


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Server is running!"}), 200


@app.route('/analyze', methods=['POST'])
def analyze_dosimeter():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # --- NEW: worker ID + shift, for DGMS/OISD-style occupational health records ---
    worker_id = request.form.get('worker_id', 'unknown')
    shift = request.form.get('shift', 'unspecified')

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

    # Edge case check: homography requires exactly 4 correspondence points
    if len(src_pts) < 4:
        return jsonify({"status": "Error", "message": "All 4 corner markers must be visible to align the wristband."}), 400

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    matrix, _ = cv2.findHomography(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (CARD_SIZE, CARD_SIZE))
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32)

    # --- Reference patches (positioned to avoid overlapping the 4 corner markers) ---
    white_roi = warped_rgb[20:60, 130:170]
    gray_roi = warped_rgb[130:170, 240:280]

    white_measured = np.mean(white_roi.reshape(-1, 3), axis=0)
    gray_measured = np.mean(gray_roi.reshape(-1, 3), axis=0)

    avg_brightness = np.mean(white_measured)
    if avg_brightness < 100:
        return jsonify({"status": "Error", "message": "Photo is too dark. Please retake in better lighting."}), 400
    if avg_brightness > 250:
        return jsonify({"status": "Error", "message": "Photo is overexposed/too bright (glare). Please retake avoiding direct light reflection."}), 400

    # --- Two-point calibration, done in LINEAR light ---
    white_target = srgb_to_linear(np.array([240.0, 240.0, 240.0]))
    gray_target = srgb_to_linear(np.array([120.0, 120.0, 120.0]))

    white_lin = srgb_to_linear(white_measured)
    gray_lin = srgb_to_linear(gray_measured)

    denom = np.where(np.abs(white_lin - gray_lin) < 1e-6, 1e-6, white_lin - gray_lin)
    a_coef = (white_target - gray_target) / denom
    b_coef = white_target - a_coef * white_lin

    full_lin = srgb_to_linear(warped_rgb)
    corrected_lin = np.clip(full_lin * a_coef + b_coef, 0, 1)
    normalized_rgb = linear_to_srgb(corrected_lin).astype(np.uint8)

    unexposed_roi = normalized_rgb[130:170, 20:60]
    avg_unexposed_rgb = np.mean(unexposed_roi, axis=(0, 1)).reshape(1, 1, 3) / 255.0
    unexposed_lab = rgb2lab(avg_unexposed_rgb)

    patch_roi = normalized_rgb[130:220, 130:220]
    avg_patch_rgb = np.mean(patch_roi, axis=(0, 1)).reshape(1, 1, 3) / 255.0
    current_lab = rgb2lab(avg_patch_rgb)

    raw_delta_e = float(deltaE_ciede2000(unexposed_lab, current_lab)[0][0])

    # --- NEW: expiry/shelf-life patch (rows230:290, cols70:140 on the 300x300 card) ---
    expiry_roi = normalized_rgb[230:290, 70:140]
    avg_expiry_rgb = np.mean(expiry_roi, axis=(0, 1)).reshape(1, 1, 3) / 255.0
    expiry_lab = rgb2lab(avg_expiry_rgb)
    expiry_reference_lab = rgb2lab(EXPIRY_REFERENCE_RGB / 255.0)
    expiry_delta_e = float(deltaE_ciede2000(expiry_reference_lab, expiry_lab)[0][0])
    badge_status = "expired" if expiry_delta_e > EXPIRY_DELTA_E_THRESHOLD else "valid"

    # --- NEW: sealed reference cell (rows230:290, cols160:230) — temp/humidity compensation ---
    sealed_roi = normalized_rgb[230:290, 160:230]
    avg_sealed_rgb = np.mean(sealed_roi, axis=(0, 1)).reshape(1, 1, 3) / 255.0
    sealed_lab = rgb2lab(avg_sealed_rgb)
    sealed_expected_lab = rgb2lab(SEALED_CELL_EXPECTED_RGB / 255.0)
    env_drift_delta_e = float(deltaE_ciede2000(sealed_expected_lab, sealed_lab)[0][0])

    # Subtract the environmental drift measured on the sealed (non-reactive) cell
    # from the main patch reading, since both patches share the same ambient
    # temperature/humidity conditions.
    delta_e = max(raw_delta_e - env_drift_delta_e * ENV_COMPENSATION_FACTOR, 0.0)

    # --- Noise floor gate: ignore tiny color shifts caused by camera/lighting noise ---
    if delta_e < NOISE_FLOOR_DELTA_E:
        delta_e = 0.0
        dose_ppm_hours = 0.0
    else:
        dose_ppm_hours = round(CAL_K * (delta_e ** CAL_P), 2)

    if dose_ppm_hours <= SAFE_LIMIT:
        safety_label = "SAFE"
        message = f"✅ SAFE: Exposure within limits ({dose_ppm_hours} / {SAFE_LIMIT} ppm·h, per ACGIH TWA)"
    elif dose_ppm_hours <= DANGER_LIMIT:
        safety_label = "MODERATE"
        message = f"⚠ MODERATE: Elevated exposure ({dose_ppm_hours} ppm·h) — monitor closely, limit further exposure."
    else:
        safety_label = "DANGER"
        message = f"🛑 DANGER: Exposure exceeds safe limit! ({dose_ppm_hours} ppm·h) — remove worker from area and seek fresh air immediately."

    if badge_status == "expired":
        message += " | ⚠ Badge expiry patch indicates this badge is expired — dose reading may not be reliable."

    return jsonify({
        "delta_e": round(delta_e, 2),
        "raw_delta_e": round(raw_delta_e, 2),
        "env_drift_delta_e": round(env_drift_delta_e, 2),
        "ppm_hours": dose_ppm_hours,
        "status": safety_label,
        "badge_status": badge_status,
        "expiry_delta_e": round(expiry_delta_e, 2),
        "safe_limit_ppm_hours": SAFE_LIMIT,
        "danger_limit_ppm_hours": DANGER_LIMIT,
        "standard_reference": STANDARD_REFERENCE,
        "calibration_reference": CALIBRATION_REFERENCE,
        "markers_detected": len(src_pts),
        "worker_id": worker_id,
        "shift": shift,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "disclaimer": "Prototype device for demonstration only. Not certified for occupational safety compliance. Always follow official workplace gas monitoring protocols."
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
