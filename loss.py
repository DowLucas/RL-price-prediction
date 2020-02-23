import matplotlib.pyplot as plt
import pandas as pd


def read_file(file="loss.csv"):
    df = pd.read_csv(open(file, "r"))
    df["loss"] = df["loss"].apply(lambda x: float(x))
    return df["loss"].tolist()


losses = read_file()

print(losses)

plt.scatter([x for x in range(len(losses))], losses)
plt.show()
