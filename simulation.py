import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import data_generators
from itertools import product


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
        self.x_1 = 0
        self.x_2 = 0
        self.x_1_history = []
        self.x_2_history = []
        for I_1, I_2 in zip(I_1_vals, I_2_vals):
            self.step(I_1, I_2)
        diff = self.x_1 - self.x_2
        return 1 if diff > 0 else 2

    def plot(self):
        plt.clf()
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
        df = pd.concat([df1, df2])
        p = sns.lineplot(data=df, x="t", y="value", hue="name")
        plt.savefig("imgs/LCA_plot.png", dpi=300)


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
        self.x_1 = 0
        self.x_2 = 0
        self.x_1_history = []
        self.x_2_history = []

        for I_1, I_2 in zip(I_1_vals, I_2_vals):
            self.step(I_1, I_2)
            diff = self.x_1 - self.x_2
            if diff > self.A:
                return 1
            elif diff < -self.A:
                return 2
        diff = self.x_1 - self.x_2
        return 1 if diff > 0 else 2

    def plot(self):
        plt.clf()

        df1 = pd.DataFrame(
            {
                "t": list(range(len(self.x_2_history))),
                "value": np.array(self.x_1_history) - np.array(self.x_2_history),
                "name": "y1=x1-x2",
            }
        )
        df2 = pd.DataFrame(
            {
                "t": list(range(len(self.x_2_history))),
                "value": np.array(self.x_2_history) - np.array(self.x_1_history),
                "name": "y2=x2-x1",
            }
        )
        df = pd.concat([df1, df2])
        p = sns.lineplot(data=df, x="t", y="value", hue="name")
        plt.savefig("imgs/BD_plot.png", dpi=300)


def run_experiment(model, num_trials, input_func):

    decisions = []

    for i in range(num_trials):
        I_1_vals, I_2_vals = input_func()
        d = model.simulate(I_1_vals, I_2_vals)
        decisions.append(d)

    decisions = np.array(decisions)
    count_1 = sum(decisions == 1)
    count_2 = sum(decisions == 2)
    print(f"Count 1: {count_1}, Count 2: {count_2}")


def run_experiment_3(imgname):
    """Compute reward rate for the LCA models with different inhibition and 
    leak parameter values. How do the optimal parameters change for early versus
    late pulses?        

    Args:
        imgname (str): path to output plot file
    """
    # leak parameters
    ks = np.linspace(0.01, 0.11, 11)

    # inhibition parameters
    betas = np.linspace(0.01, 0.11, 11)

    num_trials = 100

    # early pulses
    early_start = 10
    early_end = 20

    # late pulses
    late_start = 80
    late_end = 90

    early_reward_rates = []
    late_reward_rates = []

    for beta, k in list(product(betas, ks)):
        model = LCA(I_0=0.2, k=k, beta=beta, sigma=0.1)
        early_rewards = []
        late_rewards = []
        for _ in range(int(num_trials/2)):
            for label in [1, 2]:
                I_1_vals, I_2_vals = data_generators.labeled_pulse(
                    early_start, early_end, label)
                d = model.simulate(I_1_vals, I_2_vals)
                early_rewards.append(d == label)

                I_1_vals, I_2_vals = data_generators.labeled_pulse(
                    late_start, late_end, label)
                d = model.simulate(I_1_vals, I_2_vals)
                late_rewards.append(d == label)

        early_reward_rates.append(sum(early_rewards)/num_trials)
        late_reward_rates.append(sum(late_rewards)/num_trials)

    params = np.round(np.stack(list(product(betas, ks))), 2)
    betas = params[:, 0]
    ks = params[:, 1]
    df = pd.DataFrame({'early_reward': early_reward_rates,
                      'late_reward': late_reward_rates, 'beta': betas, 'k': ks})

    plt.clf()

    early_df = df.pivot('beta', 'k', 'early_reward')
    ax = plt.subplot(121)
    p = sns.heatmap(early_df, vmin=0, vmax=1, cbar=False, ax=ax)
    p.set_title('LCA reward rate - Early pulse')
    p.set_xlabel('Leak')
    p.set_ylabel('Inhibition')

    late_df = df.pivot('beta', 'k', 'late_reward')
    ax = plt.subplot(122)
    p = sns.heatmap(late_df, vmin=0, vmax=1, cbar_kws={
                    'label': 'reward rate'}, ax=ax)
    p.set_title('LCA reward rate - Late pulse')
    p.set_xlabel('Leak')
    p.set_ylabel('')

    plt.savefig(imgname, dpi=300)

    e_opt = params[np.argmax(early_reward_rates), :]
    l_opt = params[np.argmax(late_reward_rates), :]

    print(
        f'Optimal LCA params for early pulse: beta={e_opt[0]}, k={e_opt[1]}')
    print(
        f'Optimal LCA params for late pulse: beta={l_opt[0]}, k={l_opt[1]}')


def run_experiment_2(model, num_trials, imgname):
    """Run experiment comparing pulse location to percent of 
    decisions favoring 1 ("Left")

    Args:
        model (object): decision model to run
        num_trials (int): number of trials to run per pulse location
        imgname (str): path to output plot file
    """

    pcts = []

    for start, end in zip(range(0, 90), range(10, 100)):

        decisions = []
        for i in range(num_trials):
            I_1_vals, I_2_vals = data_generators.pulse(start, end)
            d = model.simulate(I_1_vals, I_2_vals)
            decisions.append(d)
        decisions = np.array(decisions)
        count_1 = sum(decisions == 1)
        pcts.append(count_1 / num_trials)

    plt.clf()
    p = sns.scatterplot(x=range(len(pcts)), y=pcts)
    p.set_xlabel("Left Pulse Start Location", fontsize=18)
    p.set_ylabel("Percent of Left Decisions", fontsize=18)
    plt.savefig(imgname, dpi=300)


if __name__ == "__main__":

    sns.set_style("dark")

    m1 = LCA(I_0=0.2, k=0.1, beta=0.01, sigma=0.1)
    m2 = BD(sigma=0.1, A=1)
    m3 = LCA(I_0=0.2, k=0.01, beta=0.1, sigma=0.1)

    # I_1_vals, I_2_vals = data_generators.late_pulse()

    # m1.simulate(I_1_vals, I_2_vals)
    # m1.plot()

    # m2.simulate(I_1_vals, I_2_vals)
    # m2.plot()

    # run_experiment(m1, 1000, data_generators.early_pulse)
    # run_experiment(m2, 1000, data_generators.early_pulse)

    # run_experiment_2(m1, 500, "imgs/LCA_by_pulse_high_leak.png")
    # run_experiment_2(m2, 500, "imgs/BD_by_pulse.png")
    # run_experiment_2(m3, 500, "imgs/LCA_by_pulse_high_inhibition")

    run_experiment_3('imgs/LCA_optimal')
