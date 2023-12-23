#
#


def addSuffix(path, suffix):
    '''
    Add a suffix to the first part of a path

    Parameters
    ----------
    path (str): input path
    suffix (str): suffix to be added
    '''

    path = path.split('.')
    path[0] = path[0] + suffix
    path = '.'.join(path)
    return path