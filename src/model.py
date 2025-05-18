from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class WineQualityModel:
    """
    Wrapper for a Decision Tree or Random Forest model.
    """

    def __init__(self, model_type='decision_tree', **kwargs):
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(**kwargs)
        else:
            self.model = DecisionTreeClassifier(**kwargs)

    def train(self, X_train, y_train):
        """
        Trains the decision tree model.
        """

        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predicts wine quality for given features.
        """

        return self.model.predict(X)

    def score(self, X, y):
        """
        Returns the accuracy of the model.
        """

        return self.model.score(X, y)
