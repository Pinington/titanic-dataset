# models.py

from abc import ABC, abstractmethod
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Model(ABC):

    def __init__(self):
        self._prediction_percentage = 0.0

    @property
    def prediction_percentage(self):
        return self._prediction_percentage

    @abstractmethod
    def train(self, X, y):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Return predictions."""
        pass

    def evaluate(self, X, y):
        predictions = self.predict(X)

        self._prediction_percentage = (
            np.mean(predictions == np.asarray(y)) * 100
        )

        return self._prediction_percentage

    def create_submission(self, X, passenger_ids, filename="submission.csv"):
        predictions = self.predict(X)

        submission = pd.DataFrame({
            "PassengerId": passenger_ids,
            "Survived": predictions.astype(int)
        })

        submission.to_csv(filename, index=False)



class NeuralNetwork(Model):

    HIDDEN_1 = 16
    HIDDEN_2 = 8

    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0008
    EPOCHS = 550
    THRESHOLD = 0.6

    def __init__(self):
        super().__init__()

        self.preprocessor = None
        self.network = None

        self.loss_function = None
        self.optimizer = None

        self.train_losses = []
        self.train_accuracies = []

        self.val_losses = []
        self.val_accuracies = []


    def _fit_preprocessor(self, X):
        categorical = ["Sex", "Embarked", "Title"]
        numerical = [
            "Pclass",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "FamilyTrip"
        ]

        self.preprocessor = ColumnTransformer([
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical
            ),
            (
                "numerical",
                StandardScaler(),
                numerical
            )
        ])

        return self.preprocessor.fit_transform(X)


    def _transform(self, X):
        X = self.preprocessor.transform(X)

        # ColumnTransformer may return a sparse matrix
        if hasattr(X, "toarray"):
            X = X.toarray()

        return torch.tensor(
            X,
            dtype=torch.float32
        )


    def train(self, X, y, X_val=None, y_val=None):

        # -------------------------
        # Training data
        # -------------------------

        X = self._fit_preprocessor(X)
        X = X.toarray() if hasattr(X, "toarray") else X

        X = torch.tensor(
            X,
            dtype=torch.float32
        )

        y = torch.tensor(
            np.asarray(y),
            dtype=torch.float32
        ).reshape(-1, 1)


        # -------------------------
        # Validation data
        # -------------------------

        if X_val is not None:

            X_val = self._transform(X_val)

            y_val = torch.tensor(
                np.asarray(y_val),
                dtype=torch.float32
            ).reshape(-1, 1)


        # -------------------------
        # Network
        # -------------------------

        self.network = nn.Sequential(
            nn.Linear(X.shape[1], self.HIDDEN_1),
            nn.ReLU(),

            nn.Linear(self.HIDDEN_1, self.HIDDEN_2),
            nn.ReLU(),

            nn.Linear(self.HIDDEN_2, 1),
            nn.Sigmoid()
        )


        # -------------------------
        # Loss + optimizer
        # -------------------------

        self.loss_function = nn.BCELoss()

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY
        )


        # -------------------------
        # Training loop
        # -------------------------

        for epoch in range(self.EPOCHS):

            # ---- Forward pass ----

            predictions = self.network(X)

            loss = self.loss_function(
                predictions,
                y
            )

            # ---- Backpropagation ----

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()


            # -------------------------
            # Metrics
            # -------------------------

            with torch.no_grad():

                # Recalculate after weight update
                train_predictions = self.network(X)

                train_loss = self.loss_function(
                    train_predictions,
                    y
                )

                train_classes = (
                    train_predictions >= self.THRESHOLD
                ).float()

                train_accuracy = (
                    train_classes == y
                ).float().mean()


                self.train_losses.append(
                    train_loss.item()
                )

                self.train_accuracies.append(
                    train_accuracy.item()
                )


                # -------------------------
                # Validation
                # -------------------------

                if X_val is not None:

                    val_predictions = self.network(X_val)

                    val_loss = self.loss_function(
                        val_predictions,
                        y_val
                    )

                    val_classes = (
                        val_predictions >= self.THRESHOLD
                    ).float()

                    val_accuracy = (
                        val_classes == y_val
                    ).float().mean()


                    self.val_losses.append(
                        val_loss.item()
                    )

                    self.val_accuracies.append(
                        val_accuracy.item()
                    )


    def predict(self, X):

        X = self._transform(X)

        with torch.no_grad():

            probabilities = self.network(X)

        return (
            probabilities >= 0.5
        ).numpy().astype(int).flatten()



class ID3(Model):

    MAX_DEPTH = 10
    MIN_SAMPLES = 2

    YOUNG_MAX = 18
    ADULT_MAX = 60

    def __init__(self):
        super().__init__()
        self.tree = None


    def _prepare_features(self, X):
        X = X.copy()

        # Convert Age into categories
        X["Age"] = pd.cut(
            X["Age"],
            bins=[-np.inf, self.YOUNG_MAX, self.ADULT_MAX, np.inf],
            labels=["young", "adult", "elderly"],
            right=False
        )

        # ID3 won't use Fare
        X = X.drop(columns=["Fare"])

        return X


    def train(self, X, y):
        X = self._prepare_features(X)

        self.tree = self._build_tree(X, y, depth=0)


    def predict(self, X):
        X = self._prepare_features(X)

        return np.array([
            self._predict_row(row, self.tree)
            for _, row in X.iterrows()
        ])


    def _entropy(self, y):
        # TODO
        pass


    def _information_gain(self, X, y, feature):
        # TODO
        pass


    def _build_tree(self, X, y, depth):
        # TODO
        pass


    def _predict_row(self, row, tree):
        # TODO
        pass
