

# In[1];

#Data preprocessing 
import numpy as np
import pandas as pd 
dataset = pd.read_csv('iris.data',header =None)
dataset.columns = ['sepal length','sepal width','petal length','petal width','class']
X = dataset.iloc[:, [0, 2]].values #petal & sepal length
y = dataset.iloc[:,4]


# In[2]:

#Encoding category data
from sklearn.preprocessing import LabelEncoder 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
    
print('Class labels:', np.unique(y))



# In[3]:

#plot data diagram

import matplotlib.pyplot as plt

plt.scatter(X[:50, 0], X[:50,1], marker = 's', s = 20, c='red', label='Iris-Setosa')
plt.scatter(X[50:100,0],X[50:100,1], marker = 'x', s =20, c = 'blue', label = 'Iris-Versicolour')
plt.scatter(X[100:150,0],X[100:150,1], marker = 'o', s =20, c = 'green', label ='Iris-Virginica')
plt.title('data scatter diagram')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.xlim(X[:,0].min()-1, X[:,0].max()+1) #set fig x-axis limit
plt.ylim(X[:,1].min()-1, X[:,1].max()+1) #set fig y-axis limit
plt.legend(loc='upper left')
plt.show()


# In[4];

#Splitting data to train set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# In[5]:

#Data feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# In[6]:

#Select model

'''
#Perceptron learning 
from sklearn.linear_model import Perceptron

ppn = Perceptron(max_iter = 10, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)   

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test!= y_pred).sum())


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)
'''

from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)


# In[7]:

#confusion matrix 
from sklearn.metrics import confusion_matrix

cm = np.matrix(confusion_matrix(y_test, y_pred))
A = [] 
for i in range(len(np.unique(y))):A.append([1])
A = np.matrix(A)
Accuracy = cm.diagonal() / (cm*A).T
print('Accuracy: %.2f' % (Accuracy*A / len(np.unique(y))))

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# In[8]:
#Plot result

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')
        


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('images/03_01.png', dpi=300)
plt.show()




