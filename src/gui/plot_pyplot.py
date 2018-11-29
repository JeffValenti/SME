import numpy as np
import matplotlib as mpl
from matplotlib.widgets import SpanSelector, Button
import matplotlib.pyplot as plt


from scipy.constants import c

clight = c * 1e-3

# from ..sme.sme import SME_Struct

from .plot_colors import PlotColors

fmt = PlotColors()


class MaskPlot:
    """ A plot that can be used to define the mask """

    # Controls:
    # a, d keys: Switch between segments
    # Left, Right Mouse button: Select sections to change the mask depending on current mode
    #       mode == "good/bad" : left  -> line mask
    #                            right -> bad mask
    #                            Does not override existing good line mask
    #
    #       mode == "line/cont" : left  -> line mask
    #                             right -> continuum mask
    #                             Does not change bad line mask
    # Shift key : Switch between "good/bad" and "line/cont" modes

    def __init__(self, sme, segment=0, axes=None, show=True):
        self.wave, self.spec, self.mask = sme.spectrum(return_mask=True)
        self.wmod, self.smod = sme.spectrum(syn=True)
        if self.wave is None:
            self.wave = self.wmod

        self.segment = segment
        self.nsegments = len(self.wave)
        self.wind = sme.wind
        self.mode = "line/cont"
        self.lines = sme.linelist
        self.vrad = np.atleast_1d(sme.vrad)
        self.vrad = [v if v is not None else 0 for v in self.vrad]
        if len(self.vrad) == 1:
            self.vrad = self.vrad * self.nsegments

        self.line_plot = None
        self.lock = False

        if axes is None:
            self.im = plt.subplots()[1]
        else:
            self.im = axes

        self.selector_line = SpanSelector(
            self.im,
            self.section_line_callback,
            direction="horizontal",
            useblit=True,
            button=(1,),
        )

        self.selector_cont = SpanSelector(
            self.im,
            self.section_continuum_callback,
            direction="horizontal",
            useblit=True,
            button=(3,),
        )

        self.im.figure.canvas.mpl_connect("key_press_event", self.key_event)
        self.im.callbacks.connect("xlim_changed", self.resize_event)

        ax_next = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.button_next = Button(ax_next, "-->")
        self.button_next.on_clicked(self.next_segment)

        ax_prev = plt.axes([0.7, 0.025, 0.1, 0.04])
        self.button_prev = Button(ax_prev, "<--")
        self.button_prev.on_clicked(self.previous_segment)

        self.plot()
        if show:
            plt.show()

    def resize_event(self, event):
        if self.line_plot is not None and not self.lock:
            xlim = np.array(self.im.get_xlim())
            xlim *= 1 - self.vrad[self.segment] / clight
            idx = (self.lines_segment.wlcent >= xlim[0]) & (
                self.lines_segment.wlcent <= xlim[1]
            )
            importance = self.lines_segment.depth - min(self.lines_segment.depth[idx])
            if max(importance[idx]) != 0:
                importance /= max(importance[idx])
            else:
                importance[idx] = 1

            for i in np.where(idx)[0]:
                self.line_plot[i][0].set_visible(True)
                self.line_plot[i][1].set_visible(True)
                self.line_plot[i][0].set_alpha(importance[i])
                self.line_plot[i][1].set_alpha(importance[i])

            for i in np.where(~idx)[0]:
                self.line_plot[i][0].set_visible(False)
                self.line_plot[i][1].set_visible(False)

    def key_event(self, event):
        if event.key in ["shift"]:
            if self.mode == "good/bad":
                self.mode = "line/cont"
            else:
                self.mode = "good/bad"
            print("Switch to mode: %s" % self.mode)

        if event.key in ["a", "left"]:
            self.goto_segment(self.segment - 1)
        if event.key in ["d", "right"]:
            self.goto_segment(self.segment + 1)

    def section_line_callback(self, min, max):
        mask_type = "line" if self.mode == "line/cont" else "good"
        self.section_select(min, max, mask_type)

    def section_continuum_callback(self, min, max):
        mask_type = "cont" if self.mode == "line/cont" else "bad"
        self.section_select(min, max, mask_type)

    def section_select(self, min, max, mask_type):
        print("%s %.3f - %.3f" % (mask_type, min, max))
        # find points
        idx = (self.wave[self.segment] <= max) & (self.wave[self.segment] >= min)

        # update masks
        if mask_type == "line":
            mask_value = 1
        elif mask_type == "cont":
            mask_value = 2
        elif mask_type == "bad":
            mask_value = 0
        elif mask_type == "good":
            mask_value = 1

        if mask_type in ["line", "cont"]:
            idx = idx & (self.mask[self.segment] != 0)
        if mask_type == "good":
            idx = idx & (self.mask[self.segment] == 0)
        self.mask[self.segment][idx] = mask_value

        # update plot
        self.lock = True
        self.update()
        self.lock = False

    def plot(self, update=False):
        if self.mask is not None:
            mask = self.mask[self.segment]

        if self.spec is not None and not update:
            self.im.plot(
                self.wave[self.segment],
                self.spec[self.segment],
                label="Observation",
                **fmt["Obs"],
            )

        if self.smod is not None and not update:
            self.im.plot(
                self.wmod[self.segment],
                self.smod[self.segment],
                label="Synthethic",
                **fmt["Syn"],
            )

        if self.spec is not None:
            self.fill_line = self.im.fill_between(
                self.wave[self.segment],
                0,
                self.spec[self.segment],
                where=mask == 1,
                label="Mask Line",
                **fmt["LineMask"],
            )

            m = mask == 2
            m[1:] = m[:-1] | m[1:]
            m[:-1] = m[:-1] | m[1:]
            self.fill_cont = self.im.fill_between(
                self.wave[self.segment],
                0,
                self.spec[self.segment],
                where=m,
                label="Mask Continuum",
                **fmt["ContMask"],
            )

        if self.lines is not None and not update:
            self.lock = True
            xlim = self.wave[self.segment][[0, -1]]
            xlim *= 1 - self.vrad[self.segment] / clight
            self.lines_segment = self.lines[
                (self.lines.wlcent >= xlim[0]) & (self.lines.wlcent <= xlim[1])
            ]

            importance = self.lines_segment.depth - min(self.lines_segment.depth)
            importance /= max(importance)
            self.line_plot = [[None, None] for _ in self.lines_segment]
            for i, line in enumerate(self.lines_segment):
                # if i > threshold:
                wl = line.wlcent * (1 + self.vrad[self.segment] / clight)
                self.line_plot[i][0] = self.im.text(
                    wl,
                    1.1,
                    f"{line.species} {line.wlcent:.2f}",
                    rotation="vertical",
                    horizontalalignment="right",
                    verticalalignment="top",
                    alpha=importance[i],
                )
                if self.spec is not None:
                    depth = np.interp(
                        wl, self.wave[self.segment], self.spec[self.segment]
                    )
                else:
                    depth = np.interp(
                        wl, self.wave[self.segment], self.smod[self.segment]
                    )
                self.line_plot[i][1] = self.im.vlines(
                    wl, ymin=depth, ymax=1.1, alpha=importance[i]
                )
            self.lock = False

        self.im.figure.suptitle("SME Fit\nSegment %i" % self.segment)
        self.im.set_xlabel("Wavelength [Å]")
        self.im.set_ylabel("normalized Intensity")
        self.im.set_ylim((0, 1.2))
        self.im.set_xlim(self.im.get_xlim())
        self.im.legend(loc="lower left")

        self.im.figure.canvas.draw()

    def update(self, reset_view=False):
        if not reset_view:
            xlim = self.im.get_xlim()
            ylim = self.im.get_ylim()

        # Remove filled between
        if reset_view:
            self.im.collections.clear()
        elif isinstance(self.im.collections[0], mpl.collections.PolyCollection):
            del self.im.collections[:2]
        else:
            del self.im.collections[-2:]
        # del self.im.collections[:2]
        self.plot(update=True)

        if not reset_view:
            self.im.set_xlim(xlim)
            self.im.set_ylim(ylim)

    def connect_axes(self):
        self.im.callbacks.connect("xlim_changed", self.resize_event)

    def goto_segment(self, segment):
        if segment >= 0 and segment < len(self.wave) - 1:
            self.segment = segment
            self.im.cla()
            self.plot()
            self.connect_axes()

    def next_segment(self, _=None):
        self.goto_segment(self.segment + 1)

    def previous_segment(self, _=None):
        self.goto_segment(self.segment - 1)

