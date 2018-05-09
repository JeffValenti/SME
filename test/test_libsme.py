import pytest
from sme.dll import LibSme
from sme.vald import LineList
from math import log10

def test_basic():
    libsme = LibSme('sme_synth.so.darwin.x86_64.64')
    print(libsme.SMELibraryVersion())
    libsme.InputWaveRange(5000, 6000)
    libsme.SetVWscale(2.5)
    libsme.SetH2broad()

    linelist = LineList()
    linelist.add('Fe 1', 5502.9931, 0.9582, -3.047, 7.19, -6.22, 239.249)
    linelist.add('Cr 2', 5503.5955, 4.1682, -2.117, 8.37, -6.49, 195.248)
    linelist.add('Mn 1', 5504.0000, 2.0000, -3.000, 8.00, -6.50, 200.247)

    libsme.InputLineList(linelist)
    outlist = libsme.OutputLineList()

    print(libsme.file, libsme.wfirst, libsme.wlast, libsme.vw_scale,
            libsme.H2broad)
    print(libsme.linelist)
    fmt = "  Out: {0:10.4f},{2:7.4f},{1:7.3f},{3:5.2f},{4:6.2f},{5:8.3f}"
    for i in range(len(linelist)):
        outline = [x for x in outlist[i]]
        outline[1] = log10(outline[1])
        print(fmt.format(*outline))
