import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class LearningCurvePlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Environment steps')
        self.ax.set_ylabel('Reward')
        if title is not None:
            self.ax.set_title(title)
        self.ax2 = None

    def add_curve(self, x, y, y_conf=None, label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(x, y, label=label)
            if y_conf is not None:
                self.ax.fill_between(x, np.subtract(y, y_conf), np.add(y, y_conf), alpha=0.2)
        else:
            self.ax.plot(x, y)

    def add_epsilon_curve(self, x, epsilon):
        ''' x: vector of environment steps
            epsilon: vector of epsilon values over time '''
        if self.ax2 is None:
            self.ax2 = self.ax.twinx()
            self.ax2.set_ylabel('Epsilon')  # Set label for secondary y-axis
        self.ax2.plot(x, epsilon, linestyle='--')

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label):
        self.ax.axhline(height, ls='--', c='k', label=label)

    def save(self, name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name, dpi=300)


def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y, window, poly)
