class Line:
    def __init__(self, species, wlcent, excit, gflog, gamrad, gamqst, gamvw):
        self.species = species
        self.wlcent = wlcent
        self.excit = excit
        self.gflog = gflog
        self.gamrad = gamrad
        self.gamqst = gamqst
        self.gamvw = gamvw

    def __str__(self):
        return "'{}',{:10.4f},{:7.4f},{:7.3f},{:5.2f},{:6.2f},{:8.3f}". \
                format(self.species, self.wlcent, self.excit, self.gflog, \
                self.gamrad, self.gamqst, self.gamvw)

class LineList:
    def __init__(self):
        self._lines = []

    def __len__(self):
        return len(self._lines)

    def __str__(self):
        out = []
        for line in self._lines:
            out.append(line.__str__())
        return '\n'.join(out)

    @property
    def species(self):
        return [line.species for line in self._lines]

    @property
    def wlcent(self):
        return [line.wlcent for line in self._lines]

    @property
    def excit(self):
        return [line.excit for line in self._lines]

    @property
    def gflog(self):
        return [line.gflog for line in self._lines]

    @property
    def gamrad(self):
        return [line.gamrad for line in self._lines]

    @property
    def gamqst(self):
        return [line.gamqst for line in self._lines]

    @property
    def gamvw(self):
        return [line.gamvw for line in self._lines]

    def add(self, species, wlcent, excit, gflog, gamrad, gamqst, gamvw):
        line = Line(species, wlcent, excit, gflog, gamrad, gamqst, gamvw)
        self._lines.append(line)

linelist = LineList()
linelist.add('Fe 1', 5502.9931, 0.9582, -3.047, 7.19, -6.22, 239.249)
linelist.add('Cr 2', 5503.5955, 4.1682, -2.117, 8.37, -6.49, 195.248)
linelist.add('Mn 1', 5504.0000, 2.0000, -3.000, 8.00, -6.50, 200.247)
print(linelist)
