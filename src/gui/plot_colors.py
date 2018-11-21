class PlotColors:
    def __init__(self, *args, **kwargs):
        self.fmt = {
            "Obs": {"color": "#1f77b4", "linestyle": "solid"},
            "Syn": {"color": "#ff7f0e", "linestyle": "solid", "marker": ""},
            "LineMask": {"facecolor": "#bcbd22", "alpha": 0.5},
            "ContMask": {"facecolor": "#d62728", "alpha": 0.5},
        }

    def __getitem__(self, key):
        return self.fmt[key]
