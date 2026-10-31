def readedf(filename):
    """Read an ESRF data file (measured at beamlines ID01 or ID02)

    Inputs
    ------
    filename: string
        the input file name

    Output
    ------
    the imported EDF structure in a dict. The scattering pattern is under key
    'data'.

    Notes
    -----
    Only datatype ``FloatValue`` is supported right now.
    """
    edf = header.readehf(filename)
    f = open(filename, 'rb')
    f.read(edf['EDF_HeaderSize'])  # skip header.
    if edf['DataType'] == 'FloatValue':
        dtype = np.float32
    else:
        raise NotImplementedError(
            'Not supported data type: %s' % edf['DataType'])
    edf['data'] = np.fromstring(f.read(edf['EDF_BinarySize']), dtype).reshape(
        edf['Dim_1'], edf['Dim_2'])
    return edf