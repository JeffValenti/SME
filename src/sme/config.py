from pathlib import Path


class SmeConfig:
    """Manage configuration of Spectroscopy Made Easy.
    """
    def __init__(self, path=None):
        self.path = path
        self.read()

    @property
    def path(self):
        """Get path for SME configuration file.
        """
        return self._path

    @path.setter
    def path(self, path):
        """Set path for SME configuration file. Default is ~/.sme/sme.cfg
        """
        if path:
            self._path = Path(path).expanduser()
        else:
            self._path = Path.home() / '.sme' / 'sme.cfg'

    @property
    def datadir(self):
        """Get directory containing SME data files, e.g., atmosphere grids.
        """
        return self._datadir

    @datadir.setter
    def datadir(self, datadir):
        """Set directory containing SME data files. Default is ~/.sme/data
        """
        if datadir:
            self._datadir = Path(datadir).expanduser()
        else:
            self._datadir = Path.home() / '.sme' / 'data'

    def read(self):
        """Read and parse the SME configuration file.
        """
        try:
            with open(self.path, 'r') as file:
                for linenum, rawline in enumerate(file):
                    line = rawline.strip()
                    if len(line) == 0 or line[0] == '#':
                        continue
                    key_value = line.split('=')
                    if len(key_value) != 2:
                        raise SyntaxError(
                            f'parsing line {linenum+1} of {self.path}\n'
                            f"   Contents: '{rawline.rstrip()}'\n"
                            f"   Expected: '<key> = <value>' specification")
                    key = key_value[0].strip().lower()
                    value = key_value[1].strip()
                    if key in ['datadir']:
                        setattr(self, key, value)
                    else:
                        raise SyntaxError
        except FileNotFoundError:
            print(f'Missing SME configuration file: {self.path}')
            print('You may want to run `sme config`')
            raise

