# Iris Classification Model: Machine learning model that will allow us to
# classify species of iris flowers. This application will introduce many
# rudimentary features and concepts of machine learning and is a good use case
# for this types of models.

# Use case: Botanist wants to determine the species of an iris flower based on
# characteristics of that flower. For instance attributes including petal
# length, width, etc. are  the "features" that determine the classification of a
# given iris flower.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


class IrisClassification:
    def __init__(self):
        """
        IrisClassification constructor
        """
        self.iris = load_iris()
        self.iris_features = self.iris.data.T

        self.sepal_length = self.iris_features[0]
        self.sepal_width = self.iris_features[1]
        self.petal_length = self.iris_features[2]
        self.petal_width = self.iris_features[3]

        self.sepal_length_label = self.iris.feature_names[0]
        self.sepal_width_label = self.iris.feature_names[1]
        self.petal_length_label = self.iris.feature_names[2]
        self.petal_width_label = self.iris.feature_names[3]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.iris['data'],
                                                                                self.iris['target'],
                                                                                random_state=0)
        self.knn = KNeighborsClassifier(n_neighbors=1)

    def save_scatter_plot(self, x, y, x_label, y_label, file_name='scatter_plot', file_format='png'):
        """
        :brief: Method to create and save scatter plot for x against y data with x_label and y_label labels.
                Optionally name and formal of the file can be set via file_name and file_format arguments.
        :param x: data object for X-axis
        :param y: data object for Y-axis
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :param file_name: name of the file
        :param file_format: format of the file
        """
        plt.scatter(x, y, c=self.iris.target)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig("{}.{}".format(file_name, file_format))

    def train_model(self):
        """
        :brief: Method to train model with X_train and y_train data.
        """
        self.knn.fit(self.X_train, self.y_train)

    def get_prediction(self, features):
        """
        :brief: Method to predict the iris species according to provided features
        :param features: numpy array with iris features: [sepal_length, sepal_width, petal_length, petal_width]
        :return: name of the predicted iris species
        """
        return self.iris['target_names'][int(self.knn.predict(features))]

    def get_model_accuracy(self):
        """
        :brief: Method to get the accuracy of the model
        :return: accuracy of the model
        """
        return self.knn.score(self.X_test, self.y_test)


if __name__ == "__main__":
    ic = IrisClassification()

    # scatter plot for sepal_width - sepal_length
    ic.save_scatter_plot(ic.sepal_width, ic.sepal_length, ic.sepal_width_label, ic.sepal_length_label)

    ic.train_model()

    X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
    print(ic.get_prediction(X_new))

    print(ic.get_model_accuracy())