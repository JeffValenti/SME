"""
Defines the colors to use in plotting
"""


class PlotColors:
    """ Define the colors to use in plotting """

    def __init__(self):
        self.fmt = {
            "Obs": {"color": "#1f77b4", "linestyle": "solid"},
            "Syn": {"color": "#ff7f0e", "linestyle": "solid", "marker": ""},
            "LineMask": {"facecolor": "#bcbd22", "alpha": 1},
            "ContMask": {"facecolor": "#d62728", "alpha": 1},
        }

    def __getitem__(self, key):
        return self.fmt[key]
