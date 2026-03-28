import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB


class naive_bayes_gaussian:

    def likelihood(self, x, mean, var):
        #gaussian likelihood
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent

    def calculate_priors(self, y):
        y = np.asarray(y).flatten()
        self.priors = {}
        self.classes = np.unique(y)
        for y_class in self.classes:
            self.priors[y_class] = np.mean(y == y_class)

    def fit(self, X, y):
        # Convert to numpy array, ensuring numeric dtype
        if hasattr(X, 'values'):  # DataFrame
            X = X.values.astype(float)
        else:
            X = np.asarray(X, dtype=float)
        y = np.asarray(y).flatten()
        #calculate mean and variance for each feature for each class
        self.feature_names = np.arange(X.shape[1])
        self.calculate_priors(y)
        self.parameters = {}
        for feature in self.feature_names:
            self.parameters[feature] = {}
            for y_class in self.classes:
                feature_values = X[y == y_class, feature]
                mean = np.mean(feature_values)
                var = np.var(feature_values)
                self.parameters[feature][y_class] = (mean, var)

    def predict(self, X_test):
        # Convert to numpy array, ensuring numeric dtype
        if hasattr(X_test, 'values'):  # DataFrame
            X_test = X_test.values.astype(float)
        else:
            X_test = np.asarray(X_test, dtype=float)
        #posterior for calss y_i = likelihood * prior
        y_pred = []
        for row in X_test:
            y_pred.append(self.predict_example(row))
        return np.array(y_pred)

    def predict_example(self, example):
        max_posterior = 0
        max_posterior_class = None
        for y_class in self.classes: #for each class, calculate posterior
            #multiply by prior
            posterior = self.priors[y_class]
            for feature in self.feature_names: #for each feature, calculate likelihood and multiply
                mean, var = self.parameters[feature][y_class]
                likelihood = self.likelihood(example[feature], mean, var)
                posterior *= likelihood

            if posterior > max_posterior: #maximize
                max_posterior = posterior
                max_posterior_class = y_class

        return max_posterior_class



def test_likelihood():
    nb = naive_bayes_gaussian()
    x = 1.6
    mean = 1.4
    var = 0.0067
    print(nb.likelihood(x, mean, var)) #approx 0.247

def test_calculate_priors():
    nb = naive_bayes_gaussian()
    y = np.array(['class1', 'class1', 'class2', 'class2', 'class2'])
    nb.calculate_priors(y)
    print('priors: ', nb.priors) #{'class1': 0.4, 'class2': 0.6}

def test_fit():
    nb = naive_bayes_gaussian()
    import pandas as pd
    X = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0],
        'feature2': [0.0, 2.0, 2.0, 10.0]
    })
    y = np.array(['class1', 'class1', 'class2', 'class2'])
    #class1: {feature1: [1.0, 2.0], feature2: [0.0, 2.0]}}
    #class2: {feature1: [3.0, 4.0], feature2: [2.0, 10.0]}}
    #class1: feature 1: mean=1.5, var=0.25, feature 2: mean=1.0, var=1.0
    #class2: feature 1: mean=3.5, var=0.25, feature 2: mean=6.0, var=16.0

    nb.fit(X, y)
    print('parameters: ', nb.parameters)  #feature 1: {'class1': (2.5, 0.25), 'class2': (3.5, 0.25)}, feature 2: {'class1': (1.0, 1.0), 'class2': (4.0, 16.0)}

def test_pred():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    print(type(data))
    X_train, X_test, y_train, y_test = train_test_split(data, iris.target, test_size=0.2, random_state=42)

    nb = naive_bayes_gaussian()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    #compare to sklearn's GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_sklearn = gnb.predict(X_test)

    print('similarity with sklearn: ', np.mean(y_pred == y_pred_sklearn)) #should be close to 1.0

if __name__ == "__main__":
    # test_likelihood()
    # test_calculate_priors()
    # test_fit()
    # test_pred()
    ...