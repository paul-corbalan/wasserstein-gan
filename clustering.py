import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from skimage import io, color
import seaborn as sns
from sklearn.linear_model import Perceptron, LogisticRegression
import shutil

def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png')):
            img_path = os.path.join(folder_path, filename)
            img = io.imread(img_path)
            r = np.array(img).ravel()
            if r.size == 256 : images.append(r)
    return images

human_faces_data = np.stack(load_images('humans'))
non_human_faces_data = np.stack(load_images('nohumans'))


human_labels = np.ones(human_faces_data.shape[0])
non_human_labels = np.zeros(non_human_faces_data.shape[0])

X = np.concatenate([human_faces_data, non_human_faces_data])
y = np.concatenate([human_labels, non_human_labels])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
perceptron_model = MLPClassifier(hidden_layer_sizes=(8,8,8,8), solver="lbfgs", activation="relu", max_iter=10000)


perceptron_model.fit(X_train, y_train)

y_pred = perceptron_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_rep)


conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Human', 'Human'],
            yticklabels=['Non-Human', 'Human'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle('Example Predictions')

for i in range(5):
    axes[0, i].imshow(X_test[i].reshape((8, 8, 4))[:,:,0])
    axes[0, i].set_title(f"Actual: {y_test[i]}, Predicted: {y_pred[i]}")
    axes[0, i].axis('off')

    axes[1, i].imshow(X_test[i + 5].reshape((8, 8, 4))[:,:,0])
    axes[1, i].set_title(f"Actual: {y_test[i + 5]}, Predicted: {y_pred[i + 5]}")
    axes[1, i].axis('off')

plt.show()


def predict_and_move(folder_path, model):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png')):
            img_path = os.path.join(folder_path, filename)
            img = io.imread(img_path)
            img_flat = img.reshape(1, -1)
            if img.size==256:
                # image_hsv = color.rgb2hsv(img[:,:,:])
                # average_hue = np.degrees(np.mean(image_hsv[:,:,0]))
                # pink_range = (300, 360, 0, 70)
                # is_pink = pink_range[0] <= average_hue <= pink_range[1] or pink_range[2] <= average_hue <= pink_range[3]
                if perceptron_model.predict_proba(img_flat)[0, 1] > 0.8:
                    print(f"Image {filename} is predicted as a human face and will be moved.")
                    shutil.copy(os.path.join(folder_path, filename), 'predicted_humans')


    # for filename, img in zip(*images):
    #     img_flat = img.reshape(1, -1)

    #     prediction = model.predict(img_flat)

    #     if prediction == 1:
    #         print(f"Image {filename} is predicted as a human face and will be moved.")
            
    #         shutil.copy(f"../{filename}", 'predicted_humans')



predict_and_move("..", perceptron_model)