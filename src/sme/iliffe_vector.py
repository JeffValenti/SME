import numpy as np


class Iliffe_vector:
    """
    Illiffe vectors are multidimensional (here 2D) but not necessarily rectangular
    Instead the index is a pointer to segments of a 1D array with varying sizes
    """

    def __init__(self, sizes, values=None, index=None, dtype=float):
        # sizes = size of the individual parts
        # the indices are then [0, s1, s1+s2, s1+s2+s3, ...]
        if index is None:
            self.__idx__ = np.concatenate([[0], np.cumsum(sizes, dtype=int)])
        else:
            if index[0] != 0:
                index = [0, *index]
            self.__idx__ = np.asarray(index, dtype=int)
            sizes = index[-1]
        # this stores the actual data
        if values is None:
            self.__values__ = np.zeros(np.sum(sizes), dtype=dtype)
        else:
            self.__values__ = np.asarray(values)

    def __len__(self):
        return len(self.__idx__) - 1

    def __getitem__(self, index):
        if not hasattr(index, "__len__"):
            index = (index,)

        if len(index) == 0:
            return self.__values__

        if isinstance(index, range):
            index = list(index)

        if isinstance(index, (list, np.ndarray)):
            values = [self[i] for i in index]
            sizes = [len(v) for v in values]
            # values has to be 1D
            values = np.concatenate(values)
            return Iliffe_vector(sizes, values=values)

        if isinstance(index, str):
            # This happens for example for np.recarrays
            return Iliffe_vector(
                None, index=self.__idx__, values=self.__values__[index]
            )

        if isinstance(index, Iliffe_vector):
            if not self.__equal_size__(index):
                raise ValueError("Index vector has a different shape")
            return self.__values__[index.__values__]

        if isinstance(index[0], slice):
            start = index[0].start if index[0].start is not None else 0
            stop = index[0].stop if index[0].stop is not None else len(self)
            step = index[0].step if index[0].step is not None else 1

            if stop > len(self):
                stop = len(self)

            idx = self.__idx__
            if step == 1:
                values = self.__values__[idx[start] : idx[stop]]
            else:
                values = []
                for i in range(start, stop, step):
                    values += [self.__values__[idx[i] : idx[i + 1]]]
                values = np.concatenate(values)
            sizes = np.diff(idx)[index[0]]

            return Iliffe_vector(sizes, values=values)

        if index[0] >= 0:
            i0 = self.__idx__[index[0]]
            i1 = self.__idx__[index[0] + 1]
        else:
            i0 = self.__idx__[index[0] - 1]
            i1 = self.__idx__[index[0]]
        if len(index) == 1:
            return self.__values__[i0:i1]
        if len(index) == 2:
            return self.__values__[i0:i1][index[1]]
        raise KeyError("Key must be maximum 2D")

    def __setitem__(self, index, value):
        if not hasattr(index, "__len__"):
            index = (index,)

        if isinstance(index, str):
            self.__values__[index] = value

        if len(index) == 0:
            self.__values__ = value
        elif len(index) in [1, 2]:
            i0 = self.__idx__[index[0]]
            i1 = self.__idx__[index[0] + 1]
            if len(index) == 1:
                self.__values__[i0:i1] = value
            elif len(index) == 2:
                self.__values__[i0:i1][index[1]] = value
        else:
            raise KeyError("Key must be maximum 2D")

    # Math operators
    # If both are Iliffe vectors of the same size, use element wise operations
    # Otherwise apply the operator to __values__
    def __equal_size__(self, other):
        if not isinstance(other, Iliffe_vector):
            return NotImplemented

        if self.shape[0] != other.shape[0]:
            return False

        return np.array_equal(self.__idx__, other.__idx__)

    def __operator__(self, other, operator):
        if isinstance(other, Iliffe_vector):
            if not self.__equal_size__(other):
                return NotImplemented
            other = other.__values__

        operator = getattr(self.__values__, operator)
        values = operator(other)
        if values is NotImplemented:
            return NotImplemented
        iv = Iliffe_vector(None, index=self.__idx__, values=values)
        return iv

    def __eq__(self, other):
        return self.__operator__(other, "__eq__")

    def __ne__(self, other):
        return self.__operator__(other, "__ne__")

    def __lt__(self, other):
        return self.__operator__(other, "__lt__")

    def __gt__(self, other):
        return self.__operator__(other, "__gt__")

    def __le__(self, other):
        return self.__operator__(other, "__le__")

    def __ge__(self, other):
        return self.__operator__(other, "__ge__")

    def __add__(self, other):
        return self.__operator__(other, "__add__")

    def __sub__(self, other):
        return self.__operator__(other, "__sub__")

    def __mul__(self, other):
        return self.__operator__(other, "__mul__")

    def __truediv__(self, other):
        return self.__operator__(other, "__truediv__")

    def __floordiv__(self, other):
        return self.__operator__(other, "__floordiv__")

    def __mod__(self, other):
        return self.__operator__(other, "__mod__")

    def __divmod__(self, other):
        return self.__operator__(other, "__divmod__")

    def __pow__(self, other):
        return self.__operator__(other, "__pow__")

    def __lshift__(self, other):
        return self.__operator__(other, "__lshift__")

    def __rshift__(self, other):
        return self.__operator__(other, "__rshift__")

    def __and__(self, other):
        return self.__operator__(other, "__and__")

    def __or__(self, other):
        return self.__operator__(other, "__or__")

    def __xor__(self, other):
        return self.__operator__(other, "__xor__")

    def __radd__(self, other):
        return self.__operator__(other, "__radd__")

    def __rsub__(self, other):
        return self.__operator__(other, "__rsub__")

    def __rmul__(self, other):
        return self.__operator__(other, "__rmul__")

    def __rtruediv__(self, other):
        return self.__operator__(other, "__rtruediv__")

    def __rfloordiv__(self, other):
        return self.__operator__(other, "__rfloordiv__")

    def __rmod__(self, other):
        return self.__operator__(other, "__rmod__")

    def __rdivmod__(self, other):
        return self.__operator__(other, "__rdivmod__")

    def __rpow__(self, other):
        return self.__operator__(other, "__rpow__")

    def __rlshift__(self, other):
        return self.__operator__(other, "__rlshift__")

    def __rrshift__(self, other):
        return self.__operator__(other, "__rrshift__")

    def __rand__(self, other):
        return self.__operator__(other, "__rand__")

    def __ror__(self, other):
        return self.__operator__(other, "__ror__")

    def __rxor__(self, other):
        return self.__operator__(other, "__rxor__")

    def __iadd__(self, other):
        return self.__operator__(other, "__iadd__")

    def __isub__(self, other):
        return self.__operator__(other, "__isub__")

    def __imul__(self, other):
        return self.__operator__(other, "__imul__")

    def __itruediv__(self, other):
        return self.__operator__(other, "__itruediv__")

    def __ifloordiv__(self, other):
        return self.__operator__(other, "__ifloordiv__")

    def __imod__(self, other):
        return self.__operator__(other, "__imod__")

    def __ipow__(self, other):
        return self.__operator__(other, "__ipow__")

    def __iand__(self, other):
        return self.__operator__(other, "__iand__")

    def __ior__(self, other):
        return self.__operator__(other, "__ior__")

    def __ixor__(self, other):
        return self.__operator__(other, "__ixor__")

    def __neg__(self):
        values = -self.__values__
        iv = Iliffe_vector(None, index=self.__idx__, values=values)
        return iv

    def __pos__(self):
        return self

    def __abs__(self):
        values = abs(self.__values__)
        iv = Iliffe_vector(None, index=self.__idx__, values=values)
        return iv

    def __invert__(self):
        values = ~self.__values__
        iv = Iliffe_vector(None, index=self.__idx__, values=values)
        return iv

    def __str__(self):
        s = [str(i) for i in self]
        s = str(s).replace("'", "")
        return s

    def __repr__(self):
        return f"Iliffe_vector({self.sizes}, {self.__values__})"

    def max(self):
        """ Maximum value in all segments """
        return np.max(self.__values__)

    def min(self):
        """ Minimum value in all segments """
        return np.min(self.__values__)

    @property
    def size(self):
        """int: number of elements in vector """
        return self.__idx__[-1]

    @property
    def shape(self):
        """tuple(int, list(int)): number of segments, array with size of each segment """
        return len(self), self.sizes

    @property
    def sizes(self):
        """list(int): Sizes of the different segments """
        return list(np.diff(self.__idx__))

    @property
    def ndim(self):
        """int: its always 2D """
        return 2

    @property
    def dtype(self):
        """dtype: numpy datatype of the values """
        return self.__values__.dtype

    @property
    def flat(self):
        """iter: Flat iterator through the values """
        return self.__values__.flat

    def flatten(self):
        """
        Returns a new(!) flattened version of the vector
        Values are identical to __values__ if the size
        of all segements equals the size of __values__

        Returns
        -------
        flatten: array
            new flat (1d) array of the values within this Iliffe vector
        """
        return np.concatenate([self[i] for i in range(len(self))])

    def ravel(self):
        """
        View of the contained values as a 1D array.
        Not a copy

        Returns
        -------
        raveled: array
            1d array of the contained values
        """

        return self.__values__

    def copy(self):
        """
        Create a copy of the current vector

        Returns
        -------
        copy : Iliffe_vector
            A copy of this vector
        """
        idx = np.copy(self.__idx__)
        values = np.copy(self.__values__)
        return Iliffe_vector(None, index=idx, values=values)

    def append(self, other):
        """
        Append a new segment to the end of the vector
        This creates new memory arrays for the values and the index
        """
        self.__values__ = np.concatenate((self.__values__, other))
        self.__idx__ = np.concatenate((self.__idx__, len(other)))
