from time import time
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import time
from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage.feature import hog
#START: OWN CODE
path='/Users/yeehengwang/Desktop/ML_FINAl/datasets'

lfw_dataset = sklearn.datasets.fetch_lfw_people(data_home = path, min_faces_per_person=100,  download_if_missing = False)
n_samples, h, w = lfw_dataset.images.shape
print('gray image h(height):',h)
print('gray image w(width):',w)
print('the numbers of pixels in one gray image(h*w) :',h*w)
IMG = []
IMG_HOG=[]
for i in range(n_samples):
    img=lfw_dataset.images[i]
    # print(img.shape)
    # img=cv2.resize(img,(1)
    des, hog_image = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(4, 4), block_norm= 'L2',visualize=True)
    # print(des.shape)
    # print(hog_image.shape)
    IMG.append(des)
    IMG_HOG.append(hog_image)

#data feature
X = lfw_dataset.data
n_features = X.shape[1]
#data label
y = lfw_dataset.target
target_names = lfw_dataset.target_names
# print(target_names.shape)
n_classes = target_names.shape[0]
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
#split datasets
X_train, X_test, y_train, y_test = train_test_split(IMG, y, test_size=0.25, random_state=42)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.25, random_state=42,shuffle=True)

def plot_gallery(images, titles, h, w, n_row=2, n_col=2):

    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        # plt.title(titles[i], size=12)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

plot_gallery(IMG_HOG,target_names,h, w )
plt.show()

print("Fitting the classifier to the training set")
t0 = time.time()
clf = svm.SVC(kernel = 'linear')
clf.fit(X_train, y_train)
print("done in %0.3fs" % (time.time() - t0))

# predict labels for test data
print("Predicting people's names on the test set")
t0 = time.time()
y_pred = clf.predict(X_test)
print("done in %0.3fs" % (time.time() - t0))
# compute accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print("\nAccuracy: %.2f" % accuracy + "%")
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test1, prediction_titles, h, w)

plt.show()
#END: OWN CODE