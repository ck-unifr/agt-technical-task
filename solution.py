# AGT challenge
# author: Kai Chen
# date: Feb. 2017

# The task at hand is a multi-class classification problem, for which both a training and a test (validation) set are provided as csv files, 'train.csv' and 'test.csv' accordingly.
# What we ask is that you work on this classification task by building a classifier using only the training data,
# with the goal of achieving the best performance possible on the test data, classifying as correctly as possible the 'label' variable.

# The code is tested in Python 3.5.2 and anaconda 3.


from csv import reader
import numpy as np
from sklearn import preprocessing, svm
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt



# This function is used for loading data from a CSV file
def load_csv(filename):
	"""
	Load data from csv file.
	:param filename: csv file path
	:return: data matrix
	"""
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)

	return np.array(dataset)


# This function is used to convert string values to integer values.
def str_column_to_int(dataset, column):
	"""
	For each column of a matrix, change string values to integer values.
	:param dataset: data matrix
	:param column: column index
	:return:
	"""
	values = [row[column] for row in dataset]
	unique = set(values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup


# This function is used to extract the indices of columns where the values are string.
def get_str_column_indices(row):
	"""
	Get the indices of columns which contain only string values.
	:param row: data row
	:return: column indices
	"""
	column_indices = []
	for i, value in enumerate(row):
		if not is_number(value):
			column_indices.append(i)
	return column_indices


# This function is used to test if a value is a number.
def is_number(s):
	"""
	Test if the input value is a number.
	:param s:
	:return: True or False
	"""
	return is_int(s) or is_float(s)

def is_int(s):
	"""
	Test if the input value is an integer.
	:param s: a value
	:return: True or False
	"""
	try:
		int(s)
		return True
	except ValueError:
		return False

def is_float(s):
	"""
	Test if the input value is a float.
	:param s: a value
	:return: True or False
	"""
	try:
		float(s)
		return True
	except ValueError:
		return False


# This function first loads data from a CSV file. Then it cleans the data by applying the following steps:
#	- Removing the header of the content.
#  - Removing the first and second columns, since they only contain the numbers of samples. The number is not useful for classification.
#  - Dividing data into X(features) and y(labels).
#  - Converting string values into integer values.
def load_data(train_filename, shuffle=False):
	"""
	Load data from csv file. Then clean up the data.
	:param filename: csv file path
	:return: X (features), y (labels)
	"""
	# read data from csv
	dataset = load_csv(filename)

	# remove header, the first, and second columns
	dataset = dataset[1:, 2:]

	# shuffle data
	if shuffle:
		np.random.shuffle(dataset)

	# split data into X(features) and y(labels)
	X = dataset[:, 0:len(dataset[0])-2]
	y = dataset[:, len(dataset[0])-1]

	# convert string to integer
	str_column_indices = get_str_column_indices(X[0])
	for str_column in str_column_indices:
		str_column_to_int(X, str_column)

	# convert string to float and replace missing value with NaN
	i = 0
	for i, row in enumerate(X):
		for j, value in enumerate(row):
			if(len(value) == 0):
				X[i, j] = np.nan
			else:
				X[i, j] = float(value)

	X = np.asarray(X, dtype=float)

	return X, y


# This function applies dimension reduction by using PCA.
def pca(X_train, X_test, n_components = 200):
	"""
	Feature dimensionality reduction with PCA.
	:param X_train: training features
	:param X_test: test features
	:param n_components: number or principal ccomponents
	:return: dimensionality reduced training and test features
	"""
	pca = PCA(n_components=n_components).fit(X_train)
	X_train = pca.transform(X_train)
	X_test = pca.transform(X_test)

	return X_train, X_test


# In this function, I first train an SVM on the training data, i.e., X_train (features) and y_train (labels).
# Then the trained SVM is used to predict the labels for the test data, i.e., X_test (features) and y_test (labels).
def classification(X_train, y_train, X_test, y_test, grid_search=False):
	"""
	Classification with SVM.
	:param X_train: training features
	:param y_train: training labels
	:param X_test: test features
	:param y_test: test labels
	:return: accuracy
	"""

	print('training svm ...')
	clf = svm.SVC()

	if grid_search:
		param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
		clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)

	clf = clf.fit(X_train, y_train)

	print('prediction with svm ...')
	y_pred = clf.predict(X_test)

	acc = np.sum(y_test == y_pred) / len(y_test)

	print(classification_report(y_test, y_pred))
	print(confusion_matrix(y_test, y_pred))

	return y_pred, acc

# This function first applies dimension reduction on the features with PCA.
# Then it trains an SVM on the training data.
# Finally it predict the labels of the test data with the trained SVM
def classification_pca(X_train, y_train, X_test, y_test, n_components, grid_search=False):
	"""
	Classification with SVM
	:param X_train: training features
	:param y_train: training labels
	:param X_test: test features
	:param y_test: test labels
	:param n_components: number of components
	:return:
	"""
	X_train_pca, X_test_pca = pca(X_train, X_test, n_components)
	y_pred, acc = classification(X_train_pca, y_train, X_test_pca, y_test, grid_search)

	return y_pred, acc


# The goal of this function is to obtain an optimal number of principal components in a given list.
# To find an optimal number of components, each number of components in a given list is used to reduce the dimension of features.
# Then the reduced features are used to train an SVM. And the SVM is used to predict the labels of the test data where the dimension of its features are also reduced with PCA with the specified number of components.
# Then the accuracy is computed. The number of components which achieves highest accuracy is considered as the optimal number of compoenents.
def get_n_components(X_train, y_train, X_val, y_val, n_components_list = [10, 50, 100, 200, 300]):
	"""
	Find optimal number of components of PCA.
	:param X_train: training features
	:param y_train: training labels
	:param X_val: validation features
	:param y_val: validation labels
	:param n_components_list: a list contains the number of components of PCA
	:return: the optimal number of components in the given list.
	"""
	best_acc = 0
	best_n_components = 0
	acc_list = []
	for n_components in n_components_list:
		y_pred, acc = classification_pca(X_train, y_train, X_val, y_val, n_components)
		acc_list.append(acc)
		print("number of components = %d accuracy = %f" % (n_components, acc))
		if acc > best_acc:
			best_acc = acc
			best_n_components = n_components

	plt.plot(n_components_list, acc_list)
	plt.show()

	return best_n_components


# This function splits data into two parts.
# A number of percentage is used to indicated how much data is put into the first part.
def split_data(X, y, per=0.1):
	"""
	Separated data into two parts with the specified percentage.
	:param X: features
	:param y: label
	:param per: percentage
	:return: the separated data
	"""
	length = int(len(X) * per)

	return X[0:length, :], y[0:length], X[length:, :], y[length:]


# This function selects the data whose ground truth label is in a given label list.
def get_train_data(X, y, labels=[]):
	"""
	Select data has the label in a given label list.
	:param X: features
	:param y: labels
	:param labels: a label list
	:return: selected data
	"""
	X_sub = []
	y_sub = []
	for i in range(len(X)):
		if int(y[i]) in labels:
			X_sub.append(X[i])
			y_sub.append(y[i])
	return np.array(X_sub), np.array(y_sub)


# This function selects the data whose predicted label is in a given label list.
def get_test_data(X_test, y_test, y_pred, labels=[]):
	"""
	Select data has the predicted label in a given label list.
	:param X_test: features
	:param y_test: ground truth labels
	:param y_pred: predicted labels
	:param labels: a label list
	:return: selected data
	"""
	X_sub = []
	y_sub = []
	for i in range(len(X_test)):
		if int(y_pred[i]) in labels:
			X_sub.append(X_test[i])
			y_sub.append(y_test[i])
	return np.array(X_sub), np.array(y_sub)



############################################################
# 1. Load and clean training, test, and validation data
############################################################
print('\nloading data ...')

filename = 'data/test.csv'
X_test, y_test = load_data(filename)

filename = 'data/train.csv'
X_train, y_train = load_data(filename, shuffle=True)

# split data into training and validation sets
X_train, y_train, X_val, y_val = split_data(X_train, y_train, 0.1)


# show number of samples per class
y = y_train.astype(int)
plt.hist(y.tolist(), range(min(y), max(y)+1))
plt.show()


############################################################
# 2. Replace missing data
############################################################
print('\nreplacing missing data ...')

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X_train)
X_train = imp.transform(X_train)
X_val = imp.transform(X_val)
X_test = imp.transform(X_test)


############################################################
# 3. Data standardization
############################################################
print('\nstandardization ...')

std_scale = preprocessing.StandardScaler().fit(X_train)
X_train = std_scale.transform(X_train)
X_val = std_scale.transform(X_val)
X_test = std_scale.transform(X_test)

# data for relabeling
# get a sub class of data in order to train a sub classifier
X_train_sub, y_train_sub = get_train_data(X_train, y_train, [1, 2])

# split the sub class data into training and validation sets
X_train_sub, y_train_sub, X_val_sub, y_val_sub = split_data(X_train_sub, y_train_sub, 0.1)


############################################################
# 4. Predict test data label with all features
############################################################
print('\nevaluation ...')

y_pred, acc = classification(X_train, y_train, X_test, y_test, grid_search=False)
print("accuracy on test set (all features): %f " % acc)


################################################################
# 5. Predict test data label with PCA of 99% variance retained
###############################################################
print('\nevaluation (PCA)...')
variance = 0.99

y_pred, acc = classification_pca(X_train, y_train, X_test, y_test, variance, grid_search=False)
print("accuracy on test set (PCA with %f variance retained): %f " % (variance, acc))



############################################################
# 6. Find a optimal number of principal components of PCA
############################################################
print('\nsearching optimal number of principal components ...')


n_components_list = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
n_components = get_n_components(X_train, y_train, X_val, y_val, n_components_list)
print("optimal number of components: %d " % n_components)



############################################################
# 7. Predict test data label with PCA
# on the optimal number of principal components
############################################################
print('\nevaluation ...')

y_pred, acc = classification_pca(X_train, y_train, X_test, y_test, n_components, grid_search=False)
print("accuracy on test set (PCA): %f " % acc)



###################################################################
# 8. Relabel the data whose label is in a sub label list. (Failed)
###################################################################
print('\nrelabeling ...')

X_test_sub, y_test_sub = get_test_data(X_test, y_test, y_pred, [1, 2])


n_components = get_n_components(X_train_sub, y_train_sub, X_val_sub, y_val_sub, n_components_list)
print(n_components)

y_pred_sub, acc = classification_pca(X_train_sub, y_train_sub, X_test_sub, y_test_sub, n_components, grid_search=False)
print("accuracy on sub test set (PCA): %f " % acc)


y_test_all = []
y_pred_all = []
for i in range(len(y_pred)):
	if int(y_pred[i]) not in [1, 2]:
		y_test_all.append(y_test[i])
		y_pred_all.append(y_pred[i])

for i in range(len(y_pred_sub)):
	y_test_all.append(y_test_sub[i])
	y_pred_all.append(y_pred_sub[i])

y_test_all = np.array(y_test_all)
y_pred_all = np.array(y_pred_all)

print(classification_report(y_test_all, y_pred_all))
print(confusion_matrix(y_test_all, y_pred_all))
acc = np.sum(y_test_all == y_pred_all) / len(y_test_all)
print("accuracy on test set (relabeling): %f " % acc)



############################################################
# 9. Feature selection
############################################################
print('\nfeature selection ...')

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
model = SelectFromModel(lsvc, prefit=True)
X_train = model.transform(X_train)
print(X_train.shape)
X_test = model.transform(X_test)
y_pred, acc = classification(X_train, y_train, X_test, y_test, grid_search=True)
print(acc)


"""
This script shows one solution of predicting the label of unseen data (test data) by using machine learning methods.
The data is cleaned by converting string values to integer values, replacing missing value with the mean of the column, and standardization.
With an SVM, trained on the given features, I achieve about 92% classification accuracy.
In order to reduce the dimensionality of the features, PCA is applied.
I show that by applying a optimal number of principal components searching method, with 500 features, I achieve comparable performance compared to using all the features.
While the original feature size is 564.
I also find that the data from class 1 and class 2 is similar.
The given features are not discriminative enough to separate the two classes.
Therefore, one future work is to discover more discriminative features in order to separated the two classes.
Collecting more training data may also increase the performance of the classification.
"""





