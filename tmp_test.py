# sweep_conf.py — test the rejected palm image at descending detection thresholds
import sys
from qc.checks.hand_landmarker import create_hand_landmarker, detect_hand

# Pass the image path as an argument, or hard-code it below.
img = sys.argv[1] if len(sys.argv) > 1 else "palm_204_L_N.jpg"

for conf in (0.5, 0.3, 0.1):
    det = create_hand_landmarker(num_hands=1, min_hand_detection_confidence=conf)
    try:
        r = detect_hand(img, detector=det)
    finally:
        det.close()
    lm = "21/21" if r.landmarks_norm else "0/21"
    print(f"conf={conf} -> ok={r.ok} | {r.message} | landmarks: {lm}")
    
# python tmp_test.py data\304_palm__L_N.jpg