import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Extract the raw data
df = pd.read_csv("assets/data/train.csv")

# Treating data

## First, extract titles from people's names and group them by similarity 

titles = set(["Mr.", "Mrs.", "Miss.", "Master"]) # Check docs to understand the choice of titles
title_data = []
for name in df["Name"]:
    # Cut the family name
    stripped_string = name[name.find(", ") + 2:]

    # Append the title
    title = stripped_string[:stripped_string.find(" ")]
    if title in titles:
        title_data.append(title)
    else:
        title_data.append("Other")


df["Title"] = title_data

## Then get rid of ticket numbers, names and cabins for now

del df["Cabin"]
del df["Ticket"]
del df["Name"]

## Replace missing age with title class median 

## Add wife / husband 

print(df.head())


