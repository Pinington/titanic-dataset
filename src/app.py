import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import pandas as pd
import numpy as np
from models import *

'''
This file groups data ETL as well as a main file to test models out with and 
display results, and create submission csv for the competition.
For more information about the ETL features created and the processing check the jupyter notebook in the documentation
the models chosen are in seperate files
'''


# 1 - Extract the raw data
train_df = pd.read_csv("assets/data/train.csv")
test_df = pd.read_csv("assets/data/test.csv")

# 2 - Transform the data

def transform_data(df):

    df = df.copy()

    ## 2.1 - First, extract titles from people's names and group them by similarity 

    titles = set(["Mr.", "Mrs.", "Miss.", "Master."]) # Check docs to understand the choice of titles
    title_data = []
    for name in df["Name"]:
        # Cut the family name and get title
        stripped_string = name[name.find(", ") + 2:]
        title = stripped_string[:stripped_string.find(" ")]

        # Combine different labels
        if title in ["Major.", "Sir.", "Col.", "Capt."]:
            title = "Mr."
        elif title in ["Ms.", "Mlle."]:
            title = "Miss."
        elif title in ["Mme."]:
            title = "Mrs."
        
        # Append the title
        if title in titles:
            title_data.append(title)
        else:
            title_data.append("Other")


    df["Title"] = title_data

    ## 2.2 - Then get rid of ticket numbers, names and cabins for now

    del df["Cabin"]
    del df["Ticket"]
    del df["Name"]

    ## 2.3 - Replace missing age with title class median 

    medians = df.groupby(["Title"])["Age"].median()
    means = df.groupby(["Title"])["Age"].mean()

    df.loc[df["Age"].isna(), "Age"] = (
        df.loc[df["Age"].isna(), "Title"].map(medians)
    )


    ## 2.4 - Add family trip feature

    df["FamilyTrip"] = (
        (df["SibSp"] + df["Parch"]) >= 1
    ).astype(int)

    return df


# 3 - Load and try models out
if __name__ == "__main__":

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title", "FamilyTrip"]
    target = "Survived"

    train = transform_data(train_df)
    test = transform_data(test_df)

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]

    nn = NeuralNetwork()

    nn.train(
        X_train,
        y_train
    )

    nn.create_submission(
        X_test,
        test["PassengerId"],
        "submission.csv"
    )
