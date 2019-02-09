from sme.dll import LibSme
from sme.vald import LineList, ValdShortLine
from math import log10


def test_basic():
    libsme = LibSme()
    print(libsme.SMELibraryVersion())
    libsme.InputWaveRange(5000, 6000)
    libsme.SetVWscale(2.5)
    libsme.SetH2broad()

    shorts = [
        "'Ti 1',       6554.2230,   1.4432, 1.0, -1.150, 7.870,-6.070,"
        " 284.261,  1.070, 0.606, '   9 wl:LGWSC   9 LGWSC   9 gf:LGWSC"
        "   7 K10   7 K10   7 K10  10 BPM Ti            '",
        "'MgH 1',      6556.8086,   0.9240, 1.0, -0.867, 7.060, 0.000,"
        "   0.000, 99.000, 0.021, '  12 KMGH  12 KMGH  12 KMGH"
        "  12 KMGH  12 KMGH  12 KMGH  12 KMGH (24)MgH       '"
        ]

    linelist = LineList()
    for short in shorts:
        linelist.add(ValdShortLine(short))

    libsme.InputLineList(linelist)
    outlist = libsme.OutputLineList()

    print(
        libsme.file,
        libsme.wfirst,
        libsme.wlast,
        libsme.vw_scale,
        libsme.H2broad
        )
    print(libsme.linelist)
    fmt = "  Out: {0:10.4f},{2:7.4f},{1:7.3f},{3:5.2f},{4:6.2f},{5:8.3f}"
    for i in range(len(linelist)):
        outline = [x for x in outlist[i]]
        outline[1] = log10(outline[1])
        print(fmt.format(*outline))
