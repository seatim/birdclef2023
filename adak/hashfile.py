
import os
import hashlib

from os.path import islink


def file_hash(hash_func, path, read_size=None):
    """ Compute hash of file using hash function ``hash_func``.
    """
    if read_size is None:
        read_size = 1 << 20

    state = hash_func()
    if islink(path):
        buf = os.readlink(path)
        state.update(buf)
    else:
        with open(path, 'rb') as f:
            while True:
                buf = f.read(read_size)
                if not buf:
                    break
                state.update(buf)

    return state.hexdigest()


def file_sha1(path, read_size=None):
    """ Return SHA1 hash of file.
    """
    return file_hash(hashlib.sha1, path, read_size)
