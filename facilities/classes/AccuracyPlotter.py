import numpy as np
from matplotlib import pyplot as plt


class AccuracyPlotter:
    def __init__(self, sXLabel, sYLabel, iXLimit, floatYLimit=1):
        # 初始化圖表
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlabel(sXLabel)
        self.ax.set_ylabel(sYLabel)
        self.ax.set_xlim(0, iXLimit)
        self.ax.set_ylim(0, floatYLimit)
        self.train_line, = self.ax.plot([], [], label='Training Accuracy')
        self.val_line, = self.ax.plot([], [], label='Validation Accuracy')
        self.ax.legend()
        self.fig.show()

    def update(self, floatTrainAcc, floatValidAcccc):
        train_xdata, train_ydata = self.train_line.get_data()
        train_xdata = np.append(train_xdata, len(train_xdata) + 1)
        train_ydata = np.append(train_ydata, floatTrainAcc)
        self.train_line.set_data(train_xdata, train_ydata)

        valid_xdata, valid_ydata = self.val_line.get_data()
        valid_xdata = np.append(valid_xdata, len(valid_xdata) + 1)
        valid_ydata = np.append(valid_ydata, floatValidAcccc)
        self.val_line.set_data(valid_xdata, valid_ydata)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.show()

