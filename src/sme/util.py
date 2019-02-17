def change_waveunit(wave, old, new):
    """Return list of wavelengths with new wavelength units.
    """
    oldlow = old.lower()
    newlow = new.lower()
    if newlow == oldlow:
        return wave
    factor = {
        'a': 1.0, 'nm': 10.0, 'um': 1e4, 'micron': 1e4,
        'cm-1': 1e8, '1/cm': 1e8}
    try:
        old_to_A = factor[oldlow]
        A_to_new = factor[newlow]
    except KeyError as e:
        raise ValueError(
            f"Invalid waveunit specified: old='{old}', new='{new}'\n"
            f"Valid waveunits: '" + "', '".join(factor.keys()) + "'")
    old_new = old_to_A / A_to_new
    if oldlow in ['cm-1', '1/cm']:
        try:
            return [old_new / w for w in wave]
        except TypeError:
            return old_new / wave
    elif newlow in ['cm-1', '1/cm']:
        try:
            return [1.0 / old_new / w for w in wave]
        except TypeError:
            return 1.0 / old_new / wave
    else:
        try:
            return [old_new * w for w in wave]
        except TypeError:
            return old_new * wave


def vacuum_angstrom(wave, units, medium):
    if units == 'A' or units is None:
        w = wave
    elif units == 'nm':
        w = wave * 10
    elif units == 'cm-1':
        w = 1e8 / wave
    else:
        raise ValueError("units must be 'A', 'nm', 'cm-1', or None")
    if medium == 'vac' or medium is None:
        pass
    elif medium == 'air':
        s2 = 1e8 / w / w
        n = 1.00008336624212083 + \
            0.02408926869968 / (130.1065924522 - s2) + \
            0.0001599740894897 / (38.92568793293 - s2)
        w = w * n
    else:
        raise ValueError("medium must be 'vac', 'air', or None")
    return w
