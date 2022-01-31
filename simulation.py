import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class LCA:
    def __init__(self, I_0, k, beta, sigma) -> None:
        self.I_0 = I_0
        self.k = k
        self.beta = beta
        self.sigma = sigma
        self.x_1 = 0
        self.x_2 = 0
        self.x_1_history = []
        self.x_2_history = []

    def step(self, I_1, I_2):

        self.x_1_history.append(self.x_1)
        self.x_2_history.append(self.x_2)

        delta_x_1 = (
            I_1
            - self.k * self.x_1
            - self.beta * self.x_2
            + self.I_0
            + np.random.normal(0, self.sigma)
        )

        delta_x_2 = (
            I_2
            - self.k * self.x_2
            - self.beta * self.x_1
            + self.I_0
            + np.random.normal(0, self.sigma)
        )

        self.x_1 = max(0, self.x_1 + delta_x_1)
        self.x_2 = max(0, self.x_2 + delta_x_2)

    def simulate(self, I_1_vals, I_2_vals):
        for I_1, I_2 in zip(I_1_vals, I_2_vals):
            self.step(I_1, I_2)

    def plot(self):
        df1 = pd.DataFrame(
            {
                "t": list(range(len(self.x_1_history))),
                "value": self.x_1_history,
                "name": "x1",
            }
        )
        df2 = pd.DataFrame(
            {
                "t": list(range(len(self.x_2_history))),
                "value": self.x_2_history,
                "name": "x2",
            }
        )
        df3 = pd.DataFrame(
            {
                "t": list(range(len(self.x_2_history))),
                "value": np.array(self.x_1_history) - np.array(self.x_2_history),
                "name": "x1-x2",
            }
        )
        df = pd.concat([df1, df2, df3])
        p = sns.lineplot(data=df, x="t", y="value", hue="name")
        plt.show()


class BD:
    def __init__(self, sigma, A) -> None:
        self.sigma = sigma
        self.A = A
        self.x_1 = 0
        self.x_2 = 0
        self.x_1_history = []
        self.x_2_history = []

    def step(self, I_1, I_2):

        self.x_1_history.append(self.x_1)
        self.x_2_history.append(self.x_2)

        delta_x_1 = I_1 + np.random.normal(0, self.sigma)
        delta_x_2 = I_2 + np.random.normal(0, self.sigma)

        self.x_1 = max(0, self.x_1 + delta_x_1)
        self.x_2 = max(0, self.x_2 + delta_x_2)

    def simulate(self, I_1_vals, I_2_vals):
        for I_1, I_2 in zip(I_1_vals, I_2_vals):
            self.step(I_1, I_2)

    def plot(self):
        df1 = pd.DataFrame(
            {
                "t": list(range(len(self.x_1_history))),
                "value": self.x_1_history,
                "name": "x1",
            }
        )
        df2 = pd.DataFrame(
            {
                "t": list(range(len(self.x_2_history))),
                "value": self.x_2_history,
                "name": "x2",
            }
        )
        df3 = pd.DataFrame(
            {
                "t": list(range(len(self.x_2_history))),
                "value": np.array(self.x_1_history) - np.array(self.x_2_history),
                "name": "x1-x2",
            }
        )
        df = pd.concat([df1, df2, df3])
        p = sns.lineplot(data=df, x="t", y="value", hue="name")
        plt.show()


if __name__ == "__main__":

    sns.set_style("dark")

    m1 = LCA(I_0=0.2, k=0.5, beta=0.5, sigma=0.1)

    # I_1_vals = np.linspace(0.8, 0.2, 100)
    # I_2_vals = np.linspace(0.2, 0.8, 100)
    I_1_vals = np.full((100), 0.5)
    I_2_vals = np.full((100), 0.5)
    I_1_vals[70:80] = 0.8

    m1.simulate(I_1_vals, I_2_vals)
    m1.plot()

    m2 = BD(sigma=0.1, A=0.5)
    m2.simulate(I_1_vals, I_2_vals)
    m2.plot()
