import matplotlib
matplotlib.use('TkAgg')

import cv2
import matplotlib.pyplot as plt

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# Charger l'image
image = cv2.imread("visage.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Détection avec Haar Cascade
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Boîte réelle (ground truth) que tu donnes manuellement (à adapter !)
true_box = (462, 153, 220, 320)

# Dessiner la boîte réelle
cv2.rectangle(image, (true_box[0], true_box[1]),
              (true_box[0] + true_box[2], true_box[1] + true_box[3]),
              (0, 255, 0), 2)

# Dessiner la boîte prédite et calculer l'IoU
for (x, y, w, h) in faces:
    pred_box = (x, y, w, h)
    iou = compute_iou(true_box, pred_box)

    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    print(f"IoU pour la boîte prédite : {iou:.2f}")

# Affichage
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off')
plt.title("Boîte réelle (vert) vs prédite (bleu)")
plt.show()

