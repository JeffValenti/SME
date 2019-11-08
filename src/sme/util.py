from copy import copy

def change_waveunit(wave, oldunits, newunits):
    """Return wavelengths converted to new wavelength units.

    Parameters
    ----------
    wave : float or list of floats
        Input wavelength(s) in units specified by `oldunits`.

    oldunits : str
        Input units for `wave`. Allowed values are described in the table
        below. Values are case insensitive.

    newunits : str
        Units for returned wavelength(s). Allowed values are described in
        the table below. Values are case insensitive.

    Returns
    -------
    float or list of floats
        Output wavelength(s) in units specified by `newunits`.

    Raises
    ------
    ValueError
        If either `oldunits` or `newunits` is not a recognized wavelength
        unit, as described in the table below.


    The following table defines allowed values for `newunits` and `oldunits`.

    +-----------------+---------------------+
    | Unit            | Description         |
    +=================+=====================+
    | :kbd:`'A'`      | Angstroms           |
    +-----------------+---------------------+
    | :kbd:`'nm'`     | nanometers          |
    +-----------------+---------------------+
    | :kbd:`'um'`     | micrometers         |
    +-----------------+---------------------+
    | :kbd:`'micron'` | micrometers         |
    +-----------------+---------------------+
    | :kbd:`'cm^-1'`  | inverse centimeters |
    +-----------------+---------------------+
    | :kbd:`'1/cm'`   | inverse centimeters |
    +-----------------+---------------------+

    Examples
    --------
    >>> from sme.util import change_waveunit
    >>> change_waveunit(5000, 'A', 'nm')
    500.0
    >>> change_waveunit([10000, 20000], 'cm^-1', 'a')
    [10000.0, 5000.0]
    """
    oldlow = oldunits.lower()
    newlow = newunits.lower()
    if newlow == oldlow:
        return wave
    factor = {
        'a': 1.0, 'nm': 10.0, 'um': 1e4, 'micron': 1e4,
        'cm^-1': 1e8, '1/cm': 1e8}
    try:
        old_to_A = factor[oldlow]
        A_to_new = factor[newlow]
    except KeyError as e:
        raise ValueError(
            f"invalid wavelength unit: old='{oldunits}', new='{newunits}'\n"
            f"Valid wavelength units: '" + "', '".join(factor.keys()) + "'")
    old_new = old_to_A / A_to_new
    if oldlow in ['cm^-1', '1/cm']:
        try:
            return [old_new / w for w in wave]
        except TypeError:
            return old_new / wave
    elif newlow in ['cm^-1', '1/cm']:
        try:
            return [1.0 / old_new / w for w in wave]
        except TypeError:
            return 1.0 / old_new / wave
    else:
        try:
            return [old_new * w for w in wave]
        except TypeError:
            return old_new * wave

def change_energyunit(energy, oldunits, newunits):
    """Return energies converted to new energy units.

    Parameters
    ----------
    energy : float or list of floats
        Input energy or energies in units specified by `oldunits`.

    oldunits : str
        Input units for `energy`. Allowed values are described
         in the table below. Values are case insensitive.

    newunits : str
        Units for returned energy or energies. Allowed values are
        described in the table below. Values are case insensitive.

    Returns
    -------
    float or list of floats
        Output energy or energies in units specified by `newunits`.

    Raises
    ------
    ValueError
        If either `oldunits` or `newunits` is not a recognized energy
        unit, as described in the table below.


    The following table defines allowed values for `newunits` and `oldunits`.

    +-----------------+---------------------+
    | Unit            | Description         |
    +=================+=====================+
    | :kbd:`'eV'`     | electron volts      |
    +-----------------+---------------------+
    | :kbd:`'erg'`    | erg                 |
    +-----------------+---------------------+
    | :kbd:`'J'`      | Joule               |
    +-----------------+---------------------+
    | :kbd:`'cm^-1'`  | inverse centimeters |
    +-----------------+---------------------+
    | :kbd:`'1/cm'`   | inverse centimeters |
    +-----------------+---------------------+

    Examples
    --------
    >>> from sme.util import change_energyunit
    >>> change_energyunit(2.0, 'eV', 'cm^-1')
    16131.087874
    >>> change_energyunit([5000, 20000], 'cm^-1', 'eV')
    [0.6199209921928418, 2.479683968771367]
    """
    oldlow = oldunits.lower()
    newlow = newunits.lower()
    if newlow == oldlow:
        return energy
    factor = {
        'ev': 1.0, 'erg': 1.602176634e-12, 'j': 1.602176634e-19,
        'cm^-1': 8065.543937, '1/cm': 8065.543937}
    try:
        old_to_eV = 1 / factor[oldlow]
        eV_to_new = factor[newlow]
    except KeyError as e:
        raise ValueError(
            f"invalid energy unit: old='{oldunits}', new='{newunits}'\n"
            f"Valid energy units: '" + "', '".join(factor.keys()) + "'")
    old_new = old_to_eV * eV_to_new
    try:
        return [old_new * e for e in energy]
    except TypeError:
        return old_new * energy

def air_to_vacuum(wair, units):
    """Convert wavelengths in air to wavelengths in vacuum.

    **Algorithm:** Convert input air wavelengths to Angstroms. Convert
    air wavelengths greater than 1999.3520267833621 Angstroms to vacuum
    wavelengths using the following formulae, which is used by VALD3:

    .. math::

        s &= {10^4 \over \lambda_{air}}

        n &= 1 + 0.00008336624212083
              + {0.02408926869968 \over 130.1065924522 - s^2}
              + {0.0001599740894897 \over 38.92568793293 - s^2}

        \lambda_{vac} &= n \lambda_{air}
     
    In this formula, :math:`n` is the index of refraction in air. Convert air
    wavelengths from Angstroms back to the original units before output.
    Do not convert vacuum wavelengths less than 2000 Angstroms.

    Parameters
    ----------
    wair : float or list of floats
        Input wavelength(s) in air.

    units : str
        Units of input wavelength(s), e.g. 'A', 'nm', 'cm^-1'.
        See :any:`change_waveunit` for list of allowed wavelength units.

    Returns
    -------
    float or list of floats
        Output wavelength(s) in vacuum and the same units as the input.

    Examples
    --------
    >>> from sme.util import vacuum_to_air
    >>> air_to_vacuum(4998.605522013399, 'A')
    5000
    >>> air_to_vacuum([x, y], 'nm')
    [500, 501]
    >>> air_to_vacuum(1999, 'A')
    1999
    """
    wlimit = 1999.3520267833621
    wair_a = change_waveunit(wair, units, 'A')
    try:
        sgen = (1e4 / w for w in wair_a)
        ngen = (1 + 0.00008336624212083 + 0.02408926869968 / \
            (130.1065924522 - s*s) + 0.0001599740894897 / \
            (38.92568793293 - s*s) for s in sgen)
        wvac_a = [w * n if w > wlimit else w for w, n in zip(wair_a, ngen)]
    except TypeError:
        if wair_a > wlimit:
            s = 1e4 / wair_a
            n = 1 + 0.00008336624212083 + 0.02408926869968 / \
                (130.1065924522 - s*s) + 0.0001599740894897 / \
                (38.92568793293 - s*s)
            wvac_a = wair_a * n
        else:
            wvac_a = wair_a
    wair = change_waveunit(wvac_a, 'A', units)
    return(wair)
    
def vacuum_to_air(wvac, units):
    """Convert wavelengths in vacuum to wavelengths in air.

    Convert input vacuum wavelengths to Angstroms. Convert vacuum wavelengths
    greater than 2000 Angstroms to air wavelengths using the following formulae
    from Morton (2000, ApJS, 130, 403), which is used by VALD3:

    .. math::

        s &= {10^4 \over \lambda_{vac}}

        n &= 1 + 0.0000834254 + {0.02406147 \over 130 - s^2}
              + {0.00015998 \over 38.9 - s^2}

        \lambda_{air} &= {\lambda_{vac} \over n}
     
    In this formula, :math:`n` is the index of refraction in air. Convert air
    wavelengths from Angstroms back to the original units before output.
    Do not convert vacuum wavelengths less than 2000 Angstroms.

    Parameters
    ----------
    wvac : float or list of floats
        Input wavelength(s) in vacuum.

    units : str
        Units of input wavelength(s), e.g. 'A', 'nm', 'cm^-1'.
        See :any:`change_waveunit` for list of allowed wavelength units.

    Returns
    -------
    float or list of floats
        Output wavelength(s) in air and the same units as the input.

    Examples
    --------
    >>> from sme.util import vacuum_to_air
    >>> vacuum_to_air(5000, 'A')
    4998.605522013399
    >>> vacuum_to_air([500, 501], 'nm')
    [499.86055220133994, 500.8602864587058]
    >>> vacuum_to_air(1999, 'A')
    1999
    """
    wvac_a = change_waveunit(wvac, units, 'A')
    try:
        sgen = (1e4 / w for w in wvac_a)
        ngen = (1 + 0.0000834254 + 0.02406147 / (130 - s*s) + \
            0.00015998 / (38.9 - s*s) for s in sgen)
        wair_a = [w / n if w > 2000 else w for w, n in zip(wvac_a, ngen)]
    except TypeError:
        if wvac_a > 2000:
            s = 1e4 / wvac_a
            n = 1 + 0.0000834254 + 0.02406147 / (130 - s*s) + \
                0.00015998 / (38.9 - s*s)
            wair_a = wvac_a / n
        else:
            wair_a = wvac_a
    wair = change_waveunit(wair_a, 'A', units)
    return(wair)

def vacuum_angstroms(wave, units, medium):
    """Convert wavelength(s) from input units and medium to vacuum angstroms.

    Internally, SME uses vacuum wavelengths in Angstroms. When reading
    spectra and line data into SME, SME uses this function to convert
    to vacuum wavelengths in Angstroms.

    Parameters
    ----------
    wave : float or list of floats
        Input wavelength(s) in units specified by `units` and medium
        specified by `medium`.

    units : str
        Input units for `wave`. See :any:`change_waveunit` for description
        of allowed wavelength units.

    medium : str
        Input medium for `wave`. Allowed values are described in the table
        below. Values are case insensitive.

    Returns
    -------
    float or list of floats
        Output vacuum wavelength(s) in Angstroms.

    Raises
    ------
    ValueError
        If `units` is not a recognized wavelength unit (as described in
        :any:`change_waveunit`) or if `medium` is not a recognized medium
        (as described in the table below).


    The following table defines allowed values for `medium`.

    +-----------------+---------------+
    | Medium          | Description   |
    +=================+===============+
    | :kbd:`'air'`    | Air           |
    +-----------------+---------------+
    | :kbd:`'vac'`    | Vacuum        |
    +-----------------+---------------+
    | :kbd:`'vacuum'` | Vacuum        |
    +-----------------+---------------+
    | :kbd:`'None'`   | Assume vacuum |
    +-----------------+---------------+

    Examples
    --------
    >>> from sme.util import vacuum_angstroms
    >>> vacuum_angstroms(5000, 'A', 'vac')
    5000
    >>> vacuum_angstroms([500, 501], 'nm', 'air')
    [5001.39484863807, 5011.397506813088]
    """
    wave_a = change_waveunit(wave, units, 'A')
    if medium.lower() in ['vac', 'vacuum', None]:
        wvac_a = copy(wave_a)
    elif medium.lower() == 'air':
        wvac_a = air_to_vacuum(wave_a, 'A')
    else:
        raise ValueError(
            f"invalid medium: '{medium}'\n"
            f"Valid media: 'air', 'vac', 'vacuum', or None")
    return(wvac_a)
