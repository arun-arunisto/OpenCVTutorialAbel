#import necessary packages
import os
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#preparing data
input_dir = "C:\\Users\\aruna\\PycharmProjects\\advancedComputerVisionAbel\\model_training\\datasets\\sklearn_data"
categories = ["cloudy", "rain", "shine", "sunrise"]

data = []
labels = []

for idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)

        try:
            img = resize(img, (15, 15))  # Ensure consistent resizing
        except Exception as e:
            print(f"Error resizing image {file}: {e}")
            continue  # Skip this image if there's an issue

        if img.shape != (15, 15, 3):  # Check if the image is in the expected shape (RGB)
            print(f"Unexpected shape for {file}: {img.shape}")
            continue  # Skip if the shape is not as expected

        data.append(img.flatten())  # Flatten the image
        labels.append(idx)
        print("Images Processed:", file)

data = np.asarray(data)
labels = np.asarray(labels)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

print("First Stage is completed!!")

#train/test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

#train the model
classifier = SVC()
parameters = [{'gamma':[0.01, 0.001, 0.0001], 'C':[1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)

#testing performance
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)
print(f"{str(score*100)}% of samples classified correctly")

#saving the model
pickle.dump(best_estimator, open("weather_model.p", "wb"))
print("Saved Successfully!!")
