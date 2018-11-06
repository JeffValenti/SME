import numpy as np
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt

# from ..sme.sme import SME_Struct

# Color and Linestyle settings
fmt = {
    "Obs": {"color": "tab:blue", "linestyle": "solid"},
    "Syn": {"color": "tab:orange", "linestyle": "solid", "marker": ""},
    "LineMask": {"facecolor": "tab:green", "alpha": 0.5},
    "ContMask": {"facecolor": "tab:purple", "alpha": 0.5},
}


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

    def __init__(self, sme, segment=0, axes=None):
        self.wave, self.spec, self.mask = sme.spectrum(return_mask=True)
        self.wmod, self.smod = sme.spectrum(syn=True)
        self.segment = segment
        self.wind = [0, *(sme.wind + 1)]
        self.mode = "line/cont"
        self.lines = sme.linelist

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

        self.plot()
        plt.show()

    def key_event(self, event):
        if event.key in ["shift"]:
            if self.mode == "good/bad":
                self.mode = "line/cont"
            else:
                self.mode = "good/bad"
            print("Switch to mode: %s" % self.mode)

        if event.key in ["a", "left"]:
            if self.segment > 0:
                self.segment -= 1
                self.update(reset_view=True)
        if event.key in ["d", "right"]:
            if self.segment < len(self.wave) - 1:
                self.segment += 1
                self.update(reset_view=True)

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
        self.update()

    def plot(self):
        if self.mask is not None:
            mask = self.mask[self.segment]

        if self.spec is not None:
            self.im.plot(
                self.wave[self.segment],
                self.spec[self.segment],
                label="Observation",
                **fmt["Obs"],
            )

        if self.smod is not None:
            self.im.plot(
                self.wmod[self.segment],
                self.smod[self.segment],
                label="Synthethic",
                **fmt["Syn"],
            )

        if self.spec is not None:
            self.im.fill_between(
                self.wave[self.segment],
                0,
                self.spec[self.segment],
                where=mask == 1,
                label="Mask Line",
                **fmt["LineMask"],
            )
            self.im.fill_between(
                self.wave[self.segment],
                0,
                self.spec[self.segment],
                where=mask == 2,
                label="Mask Continuum",
                **fmt["ContMask"],
            )

        self.im.figure.suptitle("SME Fit\nSegment %i" % self.segment)
        self.im.set_xlabel("Wavelength [Å]")
        self.im.set_ylabel("normalized Intensity")
        self.im.legend(loc="lower left")

        self.im.figure.canvas.draw()

    def update(self, reset_view=False):
        if not reset_view:
            xlim = self.im.get_xlim()
            ylim = self.im.get_ylim()

        self.im.clear()
        self.plot()

        if not reset_view:
            self.im.set_xlim(xlim)
            self.im.set_ylim(ylim)
