import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true], ignore_index=True)
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
df = df[["content", "label"]]

# balance = np.sum(df["label"]) / len(df["label"])
# print(balance)
# 0.47701456635039424

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=48
)


model = SentenceTransformer("all-MiniLM-L6-v2")

X_train = model.encode(
    train_df["content"].tolist(),
    batch_size=32,
    show_progress_bar=True
)

X_test = model.encode(
    test_df["content"].tolist(),
    batch_size=32,
    show_progress_bar=True
)

y_train = train_df["label"].values
y_test = test_df["label"].values

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)