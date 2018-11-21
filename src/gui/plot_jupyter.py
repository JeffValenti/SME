
import ipywidgets as widgets
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from IPython.display import display

from scipy.constants import speed_of_light

clight = speed_of_light * 1e-3

from .plot_colors import PlotColors

fmt = PlotColors()
py.init_notebook_mode()


class FitPlot:
    """ Plot the sme solve fit, as iterations pass along """

    def __init__(self, wave, spec):
        self.fig = go.FigureWidget()
        self.fig.layout["xaxis"]["title"] = "Wavelength [Å]"
        self.fig.layout["yaxis"]["title"] = "Intensity"

        self.fig.add_scatter(x=wave, y=spec, name="Observation")

    def add_synth(self, wave, synth, iteration=0):
        self.fig.add_scatter(x=wave, y=synth, name=f"Iteration {iteration}")


class FinalPlot:
    def __init__(self, sme, segment=0):
        self.sme = sme
        self.wave, self.spec, self.mask = sme.spectrum(return_mask=True)
        self.wmod, self.smod = sme.spectrum(syn=True)
        self.nsegments = len(self.wave)
        self.segment = segment
        self.wind = [0, *(sme.wind + 1)]
        self.lines = sme.linelist
        self.vrad = np.atleast_1d(sme.vrad)[-1]

        self.mask_type = "good"

        self.fig = go.FigureWidget()
        self.fig.layout["dragmode"] = "select"
        self.fig.layout["selectdirection"] = "h"
        self.fig.layout["title"] = f"Segment {segment}"
        self.fig.layout["xaxis"]["title"] = "Wavelength [Å]"
        self.fig.layout["yaxis"]["title"] = "Intensity"

        # Add buttons to switch segments
        self.button_prev = widgets.Button(description="Previous")
        self.button_prev.on_click(self.prev_segment)

        self.button_next = widgets.Button(description="Next")
        self.button_next.on_click(self.next_segment)

        self.button_mask = widgets.ToggleButtons(
            options=["Good", "Bad", "Continuum", "Line"], description="Mask"
        )
        self.button_mask.observe(self.on_toggle_click, "value")

        self.widget = widgets.VBox(
            [
                widgets.HBox([self.button_prev, self.button_next]),
                self.button_mask,
                self.fig,
            ]
        )
        display(self.widget)
        self.create_plot()

    def shift_mask(self, x, mask):
        for i in np.where(mask)[0]:
            try:
                if mask[i] == mask[i + 1]:
                    x[i] = x[i - 1]
                else:
                    x[i] = x[i + 1]
            except IndexError:
                pass

        return x

    def create_mask_points(self, x, y, mask, value):
        mask = mask != value
        x = np.copy(x)
        y = np.copy(y)
        y[mask] = 0
        x = self.shift_mask(x, mask)
        return x, y

    def create_plot(self):
        seg = self.segment

        # Line mask
        x, y = self.create_mask_points(
            self.wave[seg], self.spec[seg], self.mask[seg], 1
        )

        self.line_mask = self.fig.add_scatter(
            x=x,
            y=y,
            fillcolor=fmt["LineMask"]["facecolor"],
            fill="tozeroy",
            mode="none",
            name="Line Mask",
            hoverinfo="none",
        )

        # Cont mask
        x, y = self.create_mask_points(
            self.wave[seg], self.spec[seg], self.mask[seg], 2
        )

        self.cont_mask = self.fig.add_scatter(
            x=x,
            y=y,
            fillcolor=fmt["ContMask"]["facecolor"],
            fill="tozeroy",
            mode="none",
            name="Continuum Mask",
            hoverinfo="none",
        )

        # Observation
        self.obs = self.fig.add_scatter(
            x=self.wave[seg],
            y=self.spec[seg],
            line={"color": fmt["Obs"]["color"]},
            name="Observation",
        )

        # Synthetic, if available
        if self.smod is not None:
            self.synth = self.fig.add_scatter(
                x=self.wave[seg],
                y=self.smod[seg],
                name="Synthethic",
                line={"color": fmt["Syn"]["color"]},
            )

        # mark important lines
        if self.lines is not None:
            xlimits = self.wave[self.segment][[0, -1]]
            xlimits *= 1 - self.vrad / clight
            lines = (self.lines.wlcent > xlimits[0]) & (self.lines.wlcent < xlimits[1])
            lines = self.lines[lines]

            # Keep only the 100 stongest lines for performance
            lines.sort("depth", ascending=False)
            lines = lines[:20,]

            x = lines.wlcent * (1 + self.vrad / clight)
            y = np.interp(x, self.wave[self.segment], self.spec[self.segment])

            annotations = []
            for i, line in enumerate(lines):
                annotations += [
                    {
                        "x": x[i],
                        "y": y[i],
                        "xref": "x",
                        "yref": "y",
                        "text": f"{line.species}",
                        "hovertext": f"{line.wlcent}",
                        "textangle": 90,
                        "opacity": 1,
                        "ax": 0,
                        "ay": 1.2,
                        "ayref": "y",
                        "showarrow": True,
                        "arrowhead": 7,
                        "xanchor": "left",
                    }
                ]
            self.fig.layout.annotations = annotations

        self.obs.on_selection(self.selection_fn)

    def update(self):
        # reset data, and plot everything again
        # TODO: how to batch this together
        self.fig.data = []
        self.fig.layout.annotations = []
        self.create_plot()

    def selection_fn(self, trace, points, selector):
        xrange = selector.xrange
        wave = self.wave[self.segment]
        mask = self.mask[self.segment]

        # Choose pixels and value depending on selected type
        if self.mask_type == "good":
            value = 1
            idx = (wave > xrange[0]) & (wave < xrange[1]) & (mask == 0)
        elif self.mask_type == "bad":
            value = 0
            idx = (wave > xrange[0]) & (wave < xrange[1])
        elif self.mask_type == "line":
            value = 1
            idx = (wave > xrange[0]) & (wave < xrange[1]) & (mask != 0)
            print(np.count_nonzero(idx))
        elif self.mask_type == "cont":
            value = 2
            idx = (wave > xrange[0]) & (wave < xrange[1]) & (mask == 1)
        else:
            return

        # Apply changes if any
        if np.count_nonzero(idx) != 0:
            self.mask[self.segment][idx] = value
            self.update()

    def next_segment(self, _=None):
        self.goto_segment(self.segment + 1)

    def prev_segment(self, _=None):
        self.goto_segment(self.segment - 1)

    def goto_segment(self, segment):
        if segment > -1 and segment < self.nsegments - 1:
            self.segment = segment
            self.fig.layout["title"] = f"Segment {segment}"
            self.update()
            # Rescale to the new segment
            self.fig.layout["xaxis"]["autorange"] = True
            self.fig.layout["yaxis"]["autorange"] = True

    def on_toggle_click(self, change):
        change = change["new"]
        if change == "Good":
            self.set_mask_good()
        elif change == "Bad":
            self.set_mask_bad()
        elif change == "Continuum":
            self.set_mask_continuum()
        elif change == "Line":
            self.set_mask_line()

    def set_mask_good(self, _=None):
        self.set_mask_type("good")

    def set_mask_bad(self, _=None):
        self.set_mask_type("bad")

    def set_mask_line(self, _=None):
        self.set_mask_type("line")

    def set_mask_continuum(self, _=None):
        self.set_mask_type("cont")

    def set_mask_type(self, type):
        self.mask_type = type
        self.fig.layout["dragmode"] = "select"

    def add(self, x, y, label=""):
        self.fig.add_scatter(x=x, y=y, name=label)
