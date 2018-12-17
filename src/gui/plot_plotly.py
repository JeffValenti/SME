"""
Provide Plotting utility for Jupyter Notebook using Plot.ly
Can also be used just for Plot.ly, which will then generated html files
"""
import ipywidgets as widgets
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go

from scipy.constants import speed_of_light
from .plot_colors import PlotColors

try:
    from IPython import get_ipython
    from IPython.display import display

    cfg = get_ipython()
    in_notebook = cfg is not None
except (AttributeError, ImportError):
    in_notebook = False

clight = speed_of_light * 1e-3
fmt = PlotColors()

if in_notebook:
    py.init_notebook_mode()


class FitPlot:
    """ Plot the sme solve fit, as iterations pass along """

    def __init__(self, wave, spec):
        self.fig = go.FigureWidget()
        self.fig.layout["xaxis"]["title"] = "Wavelength [Å]"
        self.fig.layout["yaxis"]["title"] = "Intensity"

        self.fig.add_scatter(x=wave, y=spec, name="Observation")

    def add_synth(self, wave, synth, iteration=0):
        """ add a scatter plot to the plot """
        self.fig.add_scatter(x=wave, y=synth, name=f"Iteration {iteration}")


class FinalPlot:
    """ Big plot that covers everything """

    def __init__(self, sme, segment=0):
        self.sme = sme
        self.wave = sme.wave
        self.spec = sme.spec
        self.mask = sme.mask
        self.smod = sme.synth
        self.nsegments = len(self.wave)
        self.segment = segment
        self.wind = sme.wind
        self.wran = sme.wran
        self.lines = sme.linelist
        self.vrad = sme.vrad
        self.vrad = [v if v is not None else 0 for v in self.vrad]

        self.mask_type = "good"

        data, annotations = self.create_plot(self.segment)
        self.annotations = annotations

        # Add segment slider
        steps = []
        for i in range(self.nsegments):
            step = {
                "label": f"Segment {i}",
                "method": "update",
                "args": [
                    {"visible": [v == i for v in self.visible]},
                    {
                        "title": f"Segment {i}",
                        "annotations": annotations[i],
                        "xaxis": {"range": list(self.wran[i])},
                        "yaxis": {"autorange": True},
                    },
                ],
            }
            steps += [step]

        layout = {
            "dragmode": "select",
            "selectdirection": "h",
            "title": f"Segment {segment}",
            "xaxis": {"title": "Wavelength [Å]"},
            "yaxis": {"title": "Intensity"},
            "annotations": annotations[self.segment],
            "sliders": [{"active": 0, "steps": steps}],
            "legend": {"traceorder": "reversed"},
        }
        self.fig = go.FigureWidget(data=data, layout=layout)

        # add selection callback
        self.fig.data[0].on_selection(self.selection_fn)

        # Add button to save figure
        self.button_save = widgets.Button(description="Save")
        self.button_save.on_click(self.save)

        # Add buttons for Mask selection
        self.button_mask = widgets.ToggleButtons(
            options=["Good", "Bad", "Continuum", "Line"], description="Mask"
        )
        self.button_mask.observe(self.on_toggle_click, "value")

        self.widget = widgets.VBox([self.button_mask, self.button_save, self.fig])
        if in_notebook:
            display(self.widget)

    def save(self, _=None, filename="SME.html"):
        """ save plot to html file """
        self.fig.layout.dragmode = "zoom"
        py.plot(self.fig, filename=filename)

    def shift_mask(self, x, mask):
        """ shift the edges of the mask to the bottom of the plot,
        so that the mask creates a shape with straight edges """
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
        """ Creates the points that define the outer edge of the mask region """
        mask = mask != value
        x = np.copy(x)
        y = np.copy(y)
        y[mask] = 0
        x = self.shift_mask(x, mask)
        return x, y

    def create_plot(self, current_segment):
        """ Generate the plot componentes (lines and masks) and line labels """
        seg = self.segment
        annotations = {}
        visible = []
        data = []
        line_mask_idx = {}
        cont_mask_idx = {}

        for seg in range(self.nsegments):

            k = len(visible)
            line_mask_idx[seg] = k
            cont_mask_idx[seg] = k + 1

            # The order of the plots is chosen by the z order, from low to high
            # Masks should be below the spectra (so they don't hide half of the line)
            # Synthetic on top of observation, because synthetic varies less than observation
            # Annoying I know, but plotly doesn't seem to have good controls for the z order
            # Or Legend order for that matter

            if self.mask is not None:
                # Line mask
                x, y = self.create_mask_points(
                    self.wave[seg], self.spec[seg], self.mask[seg], 1
                )

                data += [
                    dict(
                        x=x,
                        y=y,
                        fillcolor=fmt["LineMask"]["facecolor"],
                        fill="tozeroy",
                        mode="none",
                        name="Line Mask",
                        hoverinfo="none",
                        legendgroup=2,
                        visible=current_segment == seg,
                    )
                ]
                visible += [seg]

                # Cont mask
                x, y = self.create_mask_points(
                    self.wave[seg], self.spec[seg], self.mask[seg], 2
                )

                data += [
                    dict(
                        x=x,
                        y=y,
                        fillcolor=fmt["ContMask"]["facecolor"],
                        fill="tozeroy",
                        mode="none",
                        name="Continuum Mask",
                        hoverinfo="none",
                        legendgroup=2,
                        visible=current_segment == seg,
                    )
                ]
                visible += [seg]

            if self.spec is not None:
                # Observation
                data += [
                    dict(
                        x=self.wave[seg],
                        y=self.spec[seg],
                        line={"color": fmt["Obs"]["color"]},
                        name="Observation",
                        legendgroup=0,
                        visible=current_segment == seg,
                    )
                ]
                visible += [seg]

            # Synthetic, if available
            if self.smod is not None:
                data += [
                    dict(
                        x=self.wave[seg],
                        y=self.smod[seg],
                        name="Synthethic",
                        line={"color": fmt["Syn"]["color"]},
                        legendgroup=1,
                        visible=current_segment == seg,
                    )
                ]
                visible += [seg]

            # mark important lines
            if self.lines is not None:
                seg_annotations = []
                xlimits = self.wave[seg][[0, -1]]
                xlimits *= 1 - self.vrad[seg] / clight
                lines = (self.lines.wlcent > xlimits[0]) & (
                    self.lines.wlcent < xlimits[1]
                )
                lines = self.lines[lines]

                # Keep only the 100 stongest lines for performance
                lines.sort("depth", ascending=False)
                lines = lines[:20,]

                x = lines.wlcent * (1 + self.vrad[seg] / clight)
                if self.spec is not None:
                    y = np.interp(x, self.wave[seg], self.spec[seg])
                else:
                    y = np.interp(x, self.wave[seg], self.smod[seg])

                for i, line in enumerate(lines):
                    seg_annotations += [
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
                annotations[seg] = seg_annotations

        self.visible = visible
        self.line_mask_idx = line_mask_idx
        self.cont_mask_idx = cont_mask_idx

        return data, annotations

    def selection_fn(self, trace, points, selector):
        """ Callback for area selection, changes the mask depending on selected mode """
        self.segment = self.fig.layout["sliders"][0].active
        seg = self.segment

        xrange = selector.xrange
        wave = self.wave[seg]
        mask = self.mask[seg]

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
            self.mask[seg][idx] = value

            with self.fig.batch_update():
                # Update Line Mask
                m = self.line_mask_idx[seg]
                x, y = self.create_mask_points(
                    self.wave[seg], self.spec[seg], self.mask[seg], 1
                )
                self.fig.data[m].x = x
                self.fig.data[m].y = y

                # Update Cont Mask
                m = self.cont_mask_idx[seg]
                x, y = self.create_mask_points(
                    self.wave[seg], self.spec[seg], self.mask[seg], 2
                )
                self.fig.data[m].x = x
                self.fig.data[m].y = y

    def on_toggle_click(self, change):
        """ Callback for mask mode selector buttons """
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
        """ Called by clicking the 'good' mask button """
        self.set_mask_type("good")

    def set_mask_bad(self, _=None):
        """ Called by clicking the 'bad' mask button """
        self.set_mask_type("bad")

    def set_mask_line(self, _=None):
        """ Called by clicking the 'line' mask button """
        self.set_mask_type("line")

    def set_mask_continuum(self, _=None):
        """ Called by clicking the 'continuum' mask button """
        self.set_mask_type("cont")

    def set_mask_type(self, type):
        """ Changes the mask change mode and chooses the current interactive tool """
        self.mask_type = type
        self.fig.layout["dragmode"] = "select"

    def add(self, x, y, label=""):
        """ adds a scatter plot to the image, and makes the necessary changes in the slider """
        self.fig.add_scatter(x=x, y=y, name=label, legendgroup=10)
        self.visible += [-1]

        # Update Sliders
        steps = []
        for i in range(self.nsegments):
            step_visible = [(v == i) or (v == -1) for v in self.visible]
            step = {
                "label": f"Segment {i}",
                "method": "update",
                "args": [
                    {"visible": step_visible},
                    {
                        "title": f"Segment {i}",
                        "annotations": self.annotations[i],
                        "xaxis": {"range": list(self.wran[i])},
                        "yaxis": {"autorange": True},
                    },
                ],
            }
            steps += [step]

        self.fig.layout["sliders"][0]["steps"] = steps
