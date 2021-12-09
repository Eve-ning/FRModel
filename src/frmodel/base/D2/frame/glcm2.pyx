"""
    GLCM in Cython
    Copyright (C) 2021  John Chang / Eve-ning

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
cimport numpy as np
cimport cython
from skimage.util import view_as_windows

from tqdm import tqdm
from libc.math cimport sqrt
from libc.math cimport isnan

from libc.math cimport NAN


cdef enum:
    HOMOGENEITY = 0
    CORRELATION = 1
    ASM = 2
    MEAN = 3
    VAR = 4

cdef class CyGLCM:
    cdef public unsigned int radius, diameter, D2, step_size, bins, invalid_value
    cdef public np.ndarray ar
    cdef public np.ndarray features
    cdef public np.ndarray glcm
    cdef public tuple pairs
    cdef public float CORR_ERROR_VAL
    cdef public char verbose

    def __init__(self, np.ndarray[double, ndim=3] ar,
                 unsigned int radius,
                 unsigned int bins,
                 char verbose=True,
                 unsigned int step_size=1,
                 pairs=('N', 'W', 'NW', 'SW')):
        self.radius = radius
        self.step_size = step_size
        self.diameter = radius * 2 + 1
        self.bins = bins
        self.invalid_value = bins + 1
        self.ar = ar
        self.CORR_ERROR_VAL = -2

        # Dimensions of the features are
        # ROW, COL, CHN, GLCM_FEATURE
        self.features = np.zeros([<unsigned int> ar.shape[0] - (step_size + radius) * 2,
                                  <unsigned int> ar.shape[1] - (step_size + radius) * 2,
                                  ar.shape[2], 5],
                                 dtype=np.double)
        self.glcm = np.zeros([bins, bins], dtype=np.double)
        self.pairs = pairs
        self.verbose = verbose

    @cython.boundscheck(True)
    @cython.wraparound(False)
    def create_glcm(self):
        # This creates the mem_views
        cdef np.ndarray[double, ndim=3] ar = self.ar
        cdef np.ndarray[double, ndim=4] features = self.features

        # With an input of the ar(float), it binarizes and outputs to ar_bin
        cdef np.ndarray[unsigned int, ndim=3] ar_bin = self._binarize(ar)

        # This is the number of channels of the array
        # E.g. if RGB, then 3.
        cdef unsigned int chs = <unsigned int>ar_bin.shape[2]

        # This initializes the progress bar wrapper
        with tqdm(total=chs * len(self.pairs), disable=not self.verbose,
                  desc=f"GLCM Progress") as pbar:

            # We can treat each channel independently of the GLCM calculation
            for ch in range(chs):

                # directions is
                # List of Directions
                # Each Direction is a tuple of original windows and offset windows
                # directions: List[Tuple[List[window_i], List[window_j]]
                directions = self._paired_windows(ar_bin[..., ch])

                for direction in directions:
                    # Each pair is a tuple
                    # Tuple of 2 offset images for GLCM calculation.
                    self._populate_glcm(direction[0], direction[1], features[:,:,ch,:])
                    pbar.update()

        # The following statements will rescale the features to [0,1]
        # To fully understand why I do this, refer to my research journal.
        # features[..., HOMOGENEITY]    /= (self.bins - 1) ** 2 # Don't think scaling is needed.

        features[features == 0] = np.nan
        features[..., MEAN]       /= self.bins - 1
        features[..., VAR]        /= (self.bins - 1) ** 2
        features[..., CORRELATION] = (features[..., CORRELATION] + len(self.pairs)) / 2
        features /= len(self.pairs)
        return features

    @cython.boundscheck(True)
    @cython.wraparound(False)
    def _populate_glcm(self,
                       np.ndarray[unsigned int, ndim=4] windows_i,
                       np.ndarray[unsigned int, ndim=4] windows_j,
                       np.ndarray[double, ndim=3] features):
        """ For each window pair, this populates a full GLCM.

        The ar would be WR, WC, CR, CC

        :param windows_i: WR WC CR CC
        :param windows_j: WR WC CR CC
        :return:
        """
        cdef unsigned int wrs = <unsigned int>windows_i.shape[0]
        cdef unsigned int wcs = <unsigned int>windows_i.shape[1]
        cdef unsigned int wr = 0;
        cdef unsigned int wc = 0;

        for wr in range(wrs):
            for wc in range(wcs):
                # For each window_i = windows_i[wr, wc], window_j = windows_j[wr, wc]
                # We want to create the glcm and put into features[wr, wc]
                self._populate_glcm_single(windows_i[wr, wc], windows_j[wr, wc], features[wr, wc])

    @cython.cdivision(True)
    @cython.boundscheck(True)
    @cython.wraparound(False)
    cdef _populate_glcm_single(self,
                              np.ndarray[unsigned int, ndim=2] window_i,
                              np.ndarray[unsigned int, ndim=2] window_j,
                              np.ndarray[double, ndim=1] features):
        """

        :param window_i: CR CC
        :param window_j: CR CC
        :param features: SINGULAR
        :return:
        """

        cdef unsigned int cr, cc
        cdef unsigned int i = 0
        cdef unsigned int j = 0

        cdef np.ndarray[double, ndim=2] glcm = self.glcm
        glcm[:] = 0

        cdef double mean_i = 0
        cdef double mean_j = 0
        cdef double var_i = 0
        cdef double var_j = 0
        cdef double std = 0
        cdef double corr_num = 0
        cdef double corr_den = 0


        # We populate the GLCM here
        for cr in range(self.diameter):
            for cc in range(self.diameter):
                i = window_i[cr, cc]
                j = window_j[cr, cc]

                # If there are any values == bin, we just abort, since it's useless data.
                if i == self.invalid_value or j == self.invalid_value:
                    return

                mean_i += <double> i
                mean_j += <double> j
                glcm[i, j] += <double> (1 / (2 * <double>(self.diameter ** 2)))
                glcm[j, i] += <double> (1 / (2 * <double>(self.diameter ** 2))) # Symmetric for ASM.

        mean_i /= self.diameter ** 2
        mean_j /= self.diameter ** 2

        # For each cell in the GLCM

        for cr in range(self.bins):
            for cc in range(self.bins):
                features[HOMOGENEITY]   += glcm[cr, cc] / (1 + <double>(i - j) ** 2)
                features[ASM]           += glcm[cr, cc] ** 2
                var_i += glcm[cr, cc] * (<double> cr - mean_i) ** 2
                var_j += glcm[cr, cc] * (<double> cc - mean_j) ** 2

        std = <double> (sqrt(var_i) * sqrt(var_j))

        if std != 0:
            for cr in range(self.bins):
                for cc in range(self.bins):
                    features[CORRELATION] += glcm[cr, cc] * (<double> cr - mean_i) * (<double> cc - mean_j) / std

        features[MEAN] += <double> ((mean_i + mean_j) / 2)
        features[VAR] += <double> ((var_i + var_j) / 2)


    @cython.boundscheck(True)
    @cython.wraparound(False)
    def _binarize(self, np.ndarray[double, ndim=3] ar) -> np.ndarray:
        """ This binarizes the 2D image by its min-max """
        if ar.max() != 0:
            nan_mask = np.isnan(ar)
            b = (((ar - np.nanmin(ar)) / np.nanmax(ar)) * (self.bins - 1)).astype(np.uint)
            # We do this so that we can detect invalid values.
            # Note that the values will span from [0, bin-1], so bin is an invalid value as we can check.
            b[nan_mask] = self.invalid_value
            return b
        else:
            ar[np.isnan(ar)] = self.invalid_value
            return ar.astype(np.uint)

    @cython.boundscheck(True)
    def _paired_windows(self, np.ndarray[unsigned int, ndim=2] ar):
        """ Creates the pair wise windows.

        Note that this has an offset issue for corner values.
        Thus it'll cropping out the external bounds.

        E.g.
        +---------+  Here, O has a lack of 5 directions, considering 8 directions
        |O        |  Thus, it'll not be considered
        |         |
        |         |
        +---------+

        +---------+  It's trivial to see why the following locations are the only
        |         |  Places where it's possible to use all 8 directions.
        | OOOOOOO |
        |         |  Even though the user may specify a set of directions that may
        +---------+  allow a larger valid bound, we will assume this scenario to
        simplify implementation!

        The algorithm here creates pair-wise by simply having the original window and
        the offset window

        +-----+          +-----+   |   +-----+          +-----+   |
        |     |          |OOO  |   |   |     |          | OOO |   |
        | OOO |   NWest  |OOO  |   |   | OOO |   North  | OOO |   |
        | OOO | <-Pair-> |OOO  |   |   | OOO | <-Pair-> | OOO |   |
        | OOO |          |     |   |   | OOO |          |     |   |
        |     |          |     |   |   |     |          |     |   |
        +-----+          +-----+   |   +-----+          +-----+   |
                                   |                              |
        ---------------------------+------------------------------+
                                   |
         +-----+          +-----+  |  Notice that if did ALL 8 directions,
         |     |          |     |  |  you will find that the algorithm will
         | OOO |   West   |OOO  |  |  have redundancy. This is the reason
         | OOO | <-Pair-> |OOO  |  |  why NE, E, SE, S are missing.
         | OOO |          |OOO  |  |
         |     |          |     |  |  Take for example
         +-----+          +-----+  |  +-----+ Point O calculates GLCM on all
                                   |  |GGG  | 8 Directions, G
        ---------------------------+  |GO#  |
                                   |  |GGG  | In particular, (O, #) will be redundant
         +-----+          +-----+  |  +-----+
         |     |          |     |  |  +-----+ Moving Point O to the right,
         | OOO |   SWest  |     |  |  | GGG |
         | OOO | <-Pair-> |OOO  |  |  | #OG | Notice that (O, #) is the same
         | OOO |          |OOO  |  |  | GGG | GLCM pair as previously.
         |     |          |OOO  |  |  +-----+
         +-----+          +-----+  |          This is because the GLCM is bi-directional

        This redundancy is accounted for in the formula below.

        So "S" is also interpreted as "N". "SE" as "NW" and so on.
        This doesn't cause duplicates.

        :param ar: The binarized array
        :return: Returns a List of Tuple[Pair A, Pair B]
        """

        ar_w = view_as_windows(ar, (self.diameter, self.diameter))
        pairs = []
        cdef int s = self.step_size
        original = ar_w[s:-s, s:-s]


        if ("N"  in self.pairs) or ("S"  in self.pairs): pairs.append((original, ar_w[:-s-s, s:-s]))
        if ("W"  in self.pairs) or ("E"  in self.pairs): pairs.append((original, ar_w[s:-s, s+s:]))
        if ("NW" in self.pairs) or ("SE" in self.pairs): pairs.append((original, ar_w[:-s-s, s+s:]))
        if ("SW" in self.pairs) or ("NE" in self.pairs): pairs.append((original, ar_w[s+s:, s+s:]))

        return pairs

# Duplication Issue
# Not Pairs
# Not Reference Issue
#