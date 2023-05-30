
import os
import hashlib

from os.path import islink


def file_hash(hash_func, path, read_size=None):
    """ Compute hash of file using hash function ``hash_func``.

    NB: the whole file will be read and hashed regardless of ``read_size``.
    ``read_size`` is an implementation detail that enables performance
    optimization.

    NB: if the file is a symbolic link, the **link target** will be hashed.
    That is, the path pointed to by the link - the path itself, not the target
    file - will be hashed.

    Args:
        hash_func (callable): hash object constructor, e.g. `hashlib.sha1`

        path (str): path to file to hash

        read_size (int or None): number of bytes to read per
            `_io.BufferedReader.read` call.  Defaults to 1048576.

    Returns:
        The hexadecimal digest value of the hash (string)

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

    Args:
        path (str): path to file

    Returns:
        The hexadecimal digest value of the hash (string)

    """
    return file_hash(hashlib.sha1, path, read_size)
