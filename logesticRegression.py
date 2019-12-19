from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


# load data from file
def load_data():
    mnist = datasets.load_digits()
    return mnist.data, mnist.target


# change 1*n label array to 10*n matrix
def one_vs_all(label):
    new_label = []
    for i in range(len(label)):
        new_label.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(len(label)):
        new_label[i][label[i]] = 1
    return new_label


# compute the probability of each binary classsifier and choose the maximum
def predict(data, LRs):
    for j in range(len(data)):
        temp = []
        for i in range(len(LRs)):
            t = LRs[i].predict_log_proba(data)
            t2 = []
            for e in t:
                t2.append(e[1])
            temp.append(t2)

    label_data = []
    for j in range(len(temp[0])):
        z = []
        for i in range(len(temp)):
            z.append(np.array(temp)[i, j])
        label_data.append(z.index(max(z)))
    return label_data


# make 10 classifier and return the maximum probable class
def one_vs_all_logistic_regression(train_set, new_label_train, test_set):
    LRs = []
    np_label_train = np.array(new_label_train)
    for i in range(10):
        LRs.append(LogisticRegression(solver="liblinear", penalty="l2").fit(X=train_set, y=np_label_train[:, i]))
    return predict(train_set, LRs), predict(test_set, LRs)


# show the image in rows and columns
def showImage(images, true_label, predicted_label, rows, columns):
    fig = plt.figure(figsize=(8, 8))
    for j in range(1, columns * rows + 1):
        img = images[j - 1]
        temp = []
        for i in range(8):
            temp.append(img[(i * 8):(i + 1) * 8])
        fig.add_subplot(rows, columns, j)
        plt.imshow(temp, cmap='gray')
        plt.title(' pre ' + str(predicted_label[j - 1]) + '/ org ' + str(true_label[j - 1]))
    plt.show()
    return


# main part

# split data to train and test
data, label = load_data()
l_train = 2 * (len(data) // 3)
[train_set, test_set] = np.split(data, [l_train])
[label_train, label_test] = np.split(label, [l_train])
# change label in the format we need
new_label_train = one_vs_all(label_train)
# find predicted label with our classifier
predict_train, predict_test = one_vs_all_logistic_regression(train_set, new_label_train, test_set)
# print the metrics and result
print("accuracy train : " + str(accuracy_score(label_train, predict_train)))
print("accuracy test : " + str(accuracy_score(label_test, predict_test)))
print("---------------------")
print("confusion matrix train : \n" + str(confusion_matrix(label_train, predict_train)))
print("---------------------")
print("confusion matrix test : \n" + str(confusion_matrix(label_test, predict_test)))
# show randomly 25 of pictures with real and predicted label
indexs = np.random.randint(len(data) // 3, size=(1, 25))
print(indexs)
pictures = [test_set[i] for i in indexs[0]]
true_pictures = [label_test[i] for i in indexs[0]]
predicted_pictures = [predict_test[i] for i in indexs[0]]
showImage(pictures, true_pictures, predicted_pictures, 5, 5)
