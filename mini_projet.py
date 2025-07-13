import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
from sklearn.preprocessing import LabelBinarizer

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]

    union_area = boxA_area + boxB_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

# Partie 1
def face_detection_and_iou(image_path, true_box):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.rectangle(image_rgb, (true_box[0], true_box[1]), (true_box[0]+true_box[2], true_box[1]+true_box[3]), (0,255,0), 2)

    for (x, y, w, h) in faces:
        pred_box = (x, y, w, h)
        iou = compute_iou(true_box, pred_box)
        print(f"IoU: {iou:.2f}")
        cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (255,0,0), 2)

    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("Boîte réelle (vert) vs prédite (bleu)")
    plt.show()

# Partie 2
def recognition_pipeline():
    lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X, y = lfw.data, lfw.target
    target_names = lfw.target_names

    # ACP
    # Précisionn globale 76%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    pca = PCA(n_components=150, whiten=True)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    clf_svm = SVC()
    clf_svm.fit(X_train_pca, y_train)
    y_pred_pca = clf_svm.predict(X_test_pca)

    print("ACP + SVM")
    print(classification_report(y_test, y_pred_pca, target_names=target_names, zero_division=0))
    #Les classes avec beaucoup d’exemples sont bien reconnues (rappel de 0.98).
    #Les classes avec peu d’images  sont moins bien classées (rappel autour de 0.4).

    #Cela montre que l’ACP extrait efficacement les informations visuelles pertinentes, et que SVM gère bien la séparation.

    # HOG + DecisionTree / RandomForest
    hog_features = [hog(img.reshape(50, 37)) for img in X]
    X_train_hog, X_test_hog, y_train, y_test = train_test_split(hog_features, y, test_size=0.3)

    clf_tree = DecisionTreeClassifier()
    clf_tree.fit(X_train_hog, y_train)
    print("HOG + Decision Tree")
    print(classification_report(y_test, clf_tree.predict(X_test_hog), target_names=target_names, zero_division=0))
    #Précision 50%
    # L’arbre de décision surapprend vite si les descripteurs sont trop nombreux
    # Trop de bruit dans les données HOG pour un arbre simple

    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train_hog, y_train)
    print("HOG + Random Forest")
    print(classification_report(y_test, clf_rf.predict(X_test_hog), target_names=target_names, zero_division=0))
    # Précision globale 65%
    # Meilleure robustesse que l'arbre de décision, car le Random Forest réduit le surapprentissage en moyennant les résultats de plusieurs arbres.


# Exemple d'exécution

face_detection_and_iou("visage.jpg", (462, 153, 220, 320))
recognition_pipeline()


# Conclusion : L'ACP est la meilleure méthode dans notre cas car 
# Elle réduit la dimensionnalité tout en conservant l'information pertinente.
# Elle améliore la performance des classifieurs en réduisant le bruit.   
# HOG est puissant, mais il faut un classifieur plus solide que Decision Tree seul.