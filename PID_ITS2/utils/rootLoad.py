#
#   

def rootLoad():
    '''
    Load ROOT in a jupyter notebook
    '''

    import sys
    import os
    ROOTSYS = os.getenv("ROOTSYS")
    sys.path.append(ROOTSYS)
    sys.path.append(os.path.join(ROOTSYS, "bin"))
    sys.path.append(os.path.join(ROOTSYS, "include"))
    sys.path.append(os.path.join(ROOTSYS, "lib"))
    import ctypes
    ctypes.cdll.LoadLibrary(os.path.join(ROOTSYS, 'lib', 'libCore.so'))
    ctypes.cdll.LoadLibrary(os.path.join(ROOTSYS, 'lib', 'libThread.so'))
    ctypes.cdll.LoadLibrary(os.path.join(ROOTSYS, 'lib', 'libTreePlayer.so'))
    print("Shared objects loaded.")
    print("Now you can use ROOT in this notebook.")