class OpacityFlags(dict):
    """Manage opacity flags, which are passed to SME external library.
    """

    def __init__(self, values='defaults'):
        """
        Parameters
        ---------
        values : str or dictionary-like object or list-like object
        """
        self._int_to_bool = (False, True)
        self._order = (
            'H', 'H2+', 'H-', 'HRay', 'He', 'He+', 'He-',
            'HeRay', 'cool', 'luke', 'hot', 'e-', 'H2Ray')
        if values == 'defaults':
            self.defaults()
        else:
            if values.keys:
                self.from_dict(values)
            else:
                self.from_list(values)

    def defaults(self):
        """Set every opacity flag to the default value, which is True.
        By default, all sources of opacity are used.
        """
        for key in self._order:
            self[key] = True

    def from_dict(self, values):
        """Set one or more opacity flags based on input dictionary-like object.
        Keys of input dictionary-like object must be valid opacity flag names.
        Values in input dictionary must be a boolean (True or False).
        """
        assert values.keys
        for key, value in values.items():
            assert key in self._order
            assert type(value) is bool
            self[key] = value

    def from_list_of_ints(self, values):
        """Set the value of the
        """
        assert len(values) <= len(self._order)
        for i, value in enumerate(values):
            assert type(value) is int
            try:
                self[self._order[i]] = self._int_to_bool[value]
            except IndexError:
                raise(
                    f'integer values for opacity flags must be 0 or 1\n'
                    f'value for {self._order[i]} opacity flag was {value}')

    def summary(self):
        """Return short string listing separately True and False opacity flags.
        """
        out = [
            'True:', *[k for k, v in self.items() if v is True],
            ', False:', *[k for k, v in self.items() if v is False]]
        return(' '.join(out))


class SmeAtmo:
    """Manage atmosphere attributes used by the SME external library.

    Attributes
    ----------
    geometry : str
        Geometry of the model atmosphere. Use 'rhox' for a plane-parallel
        geometry on a mass column scale. Use 'tau' for a plane-parallel
        geometry on a continuum optical depth scale. Use 'sph' for a
        spherical geometry on a height scale. Case insensitive.

    radius : float
        Stellar radius (in cm) base of atmosphere grid. Mandatory for
        spherical geometry. Not allowed for plane-parallel geometry.

    opacity_flags : dictionary

    External Library Notes
    ----------------------
    Data in this class yield arguments required by the InputModel() external
    function in the SME external library. Those arguments are:

    arg[0] : Number of depths in the atmosphere
    arg[1] : Reserved for future use (currently read into TEFF, but not used)
    arg[2] : Reserved for future use (currently read into GRAV, but not used)
    arg[3] : Wavelength for reference continuous opacities (used if MOTYPE=0)
    arg[4] : Type of  model atmosphere ('TAU', 'RHOX', or 'SPH')
    """
    def __init__(self):
        print('placeholder')
