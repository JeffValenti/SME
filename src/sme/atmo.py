class OpacityFlags(dict):
    """Manage opacity flags used by the SME external library

    Attributes
    ----------
    """

    def __init__(self, values='defaults'):
        if values == 'defaults':
            self.defaults()

    def defaults(self):
        self['H'] = True
        self['H2+'] = True
        self['H-'] = True
        self['HRay'] = True
        self['He'] = True
        self['He+'] = True
        self['He-'] = True
        self['HeRay'] = True
        self['cool'] = True
        self['luke'] = True
        self['hot'] = True
        self['e-'] = True
        self['H2Ray'] = True

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
