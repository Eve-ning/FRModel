from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D


class _Frame2DChannelSpec(ABC):
    """ This is a separate class to handle spectral channels, so as to not clutter the main class too much"""

    def RED_EDGE(self: "Frame2D"): return self[..., self.CHN.RED_EDGE]
    def NIR(self: "Frame2D"):      return self[..., self.CHN.NIR]
    def NDVI(self: "Frame2D"):     return self._default(self.CHN.NDVI, self._NDVI, self.CHN.NDVI)
    def BNDVI(self: "Frame2D"):    return self._default(self.CHN.BNDVI, self._BNDVI, self.CHN.BNDVI)
    def GNDVI(self: "Frame2D"):    return self._default(self.CHN.GNDVI, self._GNDVI, self.CHN.GNDVI)
    def GARI(self: "Frame2D"):     return self._default(self.CHN.GARI, self._GARI, self.CHN.GARI)
    def GLI(self: "Frame2D"):      return self._default(self.CHN.GLI, self._GLI, self.CHN.GLI)
    def GBNDVI(self: "Frame2D"):   return self._default(self.CHN.GBNDVI, self._GBNDVI, self.CHN.GBNDVI)
    def GRNDVI(self: "Frame2D"):   return self._default(self.CHN.GRNDVI, self._GRNDVI, self.CHN.GRNDVI)
    def NDRE(self: "Frame2D"):     return self._default(self.CHN.NDRE, self._NDRE, self.CHN.NDRE)
    def LCI(self: "Frame2D"):      return self._default(self.CHN.LCI, self._LCI, self.CHN.LCI)
    def MSAVI(self: "Frame2D"):    return self._default(self.CHN.MSAVI, self._MSAVI, self.CHN.MSAVI)
    def OSAVI(self: "Frame2D"):    return self._default(self.CHN.OSAVI, self._OSAVI, self.CHN.OSAVI)

    def _NDVI(self: 'Frame2D') -> np.ndarray:
        """ Normalized Difference Vegetation Index """
        return (self.NIR().data - self.NB_RED().data) / (self.NIR().data + self.NB_RED().data)
    def _BNDVI(self: 'Frame2D') -> np.ndarray:
        """ NB_BLUE Normalized Difference Vegetation Index """
        return (self.NIR().data - self.NB_BLUE().data) / (self.NIR().data + self.NB_BLUE().data)
    def _GNDVI(self: 'Frame2D') -> np.ndarray:
        """ NB_GREEN Normalized Difference Vegetation Index """
        return (self.NIR().data - self.NB_GREEN().data) / (self.NIR().data + self.NB_GREEN().data)
    def _GARI(self: 'Frame2D') -> np.ndarray:
        """ NB_GREEN Atmospherically Resistant Vegetation Index """
        b_r = self.NB_BLUE().data - self.NB_RED().data
        return (self.NIR().data - (self.NB_GREEN().data - b_r)) / (self.NIR().data - (self.NB_GREEN().data + b_r))
    def _GLI(self: 'Frame2D') -> np.ndarray:
        """ NB_GREEN Leaf Index """
        return (2 * self.NB_GREEN().data - self.NB_RED().data - self.NB_BLUE().data) /\
               (2 * self.NB_GREEN().data + self.NB_RED().data + self.NB_BLUE().data)
    def _GBNDVI(self: 'Frame2D') -> np.ndarray:
        """ NB_GREEN NB_BLUE NDVI """
        return (self.NIR().data - self.NB_BLUE().data) / (self.NIR().data + self.NB_BLUE().data)
    def _GRNDVI(self: 'Frame2D') -> np.ndarray:
        """ NB_GREEN NB_RED NDVI """
        return (self.NIR().data - self.NB_GREEN().data) / (self.NIR().data + self.NB_GREEN().data)
    def _NDRE(self: 'Frame2D') -> np.ndarray:
        """ Normalized Difference NB_RED Edge """
        return (self.NIR().data - self.RED_EDGE().data) / (self.NIR().data + self.RED_EDGE().data)
    def _LCI(self: 'Frame2D') -> np.ndarray:
        """ Leaf Chlorophyll Index  """
        return (self.NIR().data - self.RED_EDGE().data) / (self.NIR().data + self.NB_RED().data)
    def _MSAVI(self: 'Frame2D') -> np.ndarray:
        """ Modified Soil Adjusted Vegetation Index """
        aux = (2 * self.NIR().data + 1)
        return (aux - np.sqrt(aux ** 2 - 8 * (self.NIR().data - self.NB_RED().data))) / 2
    def _OSAVI(self: 'Frame2D') -> np.ndarray:
        """ Optimized Soil Adjusted Vegetation Index """
        return (self.NIR().data - self.NB_RED().data) / (self.NIR().data + self.NB_RED().data + 0.16)
