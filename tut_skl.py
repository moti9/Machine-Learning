from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

model = DecisionTreeClassifier()
model.fit(iris.data, iris.target)

print(model)

expected = iris.target
predicted = model.predict(iris.data)

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))