''' Default data types.
'''

import numpy as np
import ctypes


class float32(np.float32):
    ctype = ctypes.c_float
    memsize = 32

    name = 'float32'
    key = 'f'

    def cstruct(self):
        return self.ctype(self)

    @staticmethod
    def ndpointer(ndim, flags):
        return np.ctypeslib.ndpointer(ndim=ndim, dtype=np.float32, flags=flags)


class float64(np.float64):
    ctype = ctypes.c_double
    memsize = 64

    name = 'float64'
    key = 'd'

    def cstruct(self):
        return self.ctype(self)

    @staticmethod
    def ndpointer(ndim, flags):
        return np.ctypeslib.ndpointer(ndim=ndim, dtype=np.float64, flags=flags)


class int16(np.int16):
    ctype = ctypes.c_int16
    memsize = 16

    name = 'int16'
    key = 's'

    def cstruct(self):
        return self.ctype(self)

    @staticmethod
    def ndpointer(ndim, flags):
        return np.ctypeslib.ndpointer(ndim=ndim, dtype=np.int16, flags=flags)


class uint16(np.uint16):
    ctype = ctypes.c_uint16
    memsize = 16

    name = 'uint16'
    key = 'us'

    def cstruct(self):
        return self.ctype(self)

    @staticmethod
    def ndpointer(ndim, flags):
        return np.ctypeslib.ndpointer(ndim=ndim, dtype=np.uint16, flags=flags)


class int32(np.int32):
    ctype = ctypes.c_int32
    memsize = 32

    name = 'int32'
    key = 'i'

    def cstruct(self):
        return self.ctype(self)

    @staticmethod
    def ndpointer(ndim, flags):
        return np.ctypeslib.ndpointer(ndim=ndim, dtype=np.int32, flags=flags)


class uint32(np.uint32):
    ctype = ctypes.c_uint32
    memsize = 32

    name = 'uint32'
    key = 'ui'

    def cstruct(self):
        return self.ctype(self)

    @staticmethod
    def ndpointer(ndim, flags):
        return np.ctypeslib.ndpointer(ndim=ndim, dtype=np.uint32, flags=flags)


class int64(np.int64):
    ctype = ctypes.c_int64
    memsize = 64

    name = 'int64'
    key = 'l'

    def cstruct(self):
        return self.ctype(self)

    @staticmethod
    def ndpointer(ndim, flags):
        return np.ctypeslib.ndpointer(ndim=ndim, dtype=np.int64, flags=flags)


class uint64(np.uint64):
    ctype = ctypes.c_uint64
    memsize = 64

    name = 'uint64'
    key = 'ul'

    def cstruct(self):
        return self.ctype(self)

    @staticmethod
    def ndpointer(ndim, flags):
        return np.ctypeslib.ndpointer(ndim=ndim, dtype=np.uint64, flags=flags)


class complex64(np.complex64):
    ctype = (ctypes.c_float, ctypes.c_float)
    memsize = 64

    name = 'complex64'
    key = 'cs'

    def cstruct(self):
        real = self.ctype[0](self.real)
        imag = self.ctype[1](self.imag)
        return real, imag

    @staticmethod
    def ndpointer(ndim, flags):
        return np.ctypeslib.ndpointer(ndim=ndim, dtype=np.complex64, flags=flags)


class complex128(np.complex128):
    ctype = (ctypes.c_double, ctypes.c_double)
    memsize = 128

    name = 'complex128'
    key = 'cd'

    def cstruct(self):
        real = self.ctype[0](self.real)
        imag = self.ctype[1](self.imag)
        return real, imag

    @staticmethod
    def ndpointer(ndim, flags):
        return np.ctypeslib.ndpointer(ndim=ndim, dtype=np.complex128, flags=flags)


class char(np.byte):
    ctype = ctypes.c_char
    memsize = 1

    name = 'char'
    key = 'c'

    def cstruct(self):
        return self.ctype(self)

    @staticmethod
    def ndpointer(ndim, flags):
        return np.ctypeslib.ndpointer(ndim=ndim, dtype=np.byte, flags=flags)


_list = [float32, float64, int16, uint16, int32, uint32, int64, uint64, complex64, complex128, char]

types = {}
types.update({ obj.name: obj for obj in _list })
types.update({ obj.key: obj for obj in _list })

