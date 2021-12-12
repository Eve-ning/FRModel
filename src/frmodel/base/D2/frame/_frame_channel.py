from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Iterable, Callable, List

import numpy as np
from skimage.color import rgb2hsv

from frmodel.base.D2.frame._frame_channel_fast_glcm import _Frame2DChannelFastGLCM
from frmodel.base.D2.frame._frame_channel_spec import _Frame2DChannelSpec

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D


class _Frame2DChannel(_Frame2DChannelFastGLCM, _Frame2DChannelSpec):

    def get_chns(self: 'Frame2D',
                 chns: Iterable[Frame2D.CHN, str] = ()) -> 'Frame2D':
        """ Gets selected channels

        Use _get_all_chns to get all but selected ones

        :param chns: Channels, can be also in strings.
        :returns Frame2D: Itself as a reference.
        """

        chns = [chns] if isinstance(chns, str) else chns

        for chn in chns:
            self.__getattribute__(chn)()

        return self

    def _default(self: "Frame2D", chns,
                 default_fn: Callable,
                 default_labels: List[str]):
        default_labels = [default_labels] if isinstance(default_labels, str) else default_labels

        try:
            return self[..., chns]
        except KeyError:
            self._data = np.append(self.data, default_fn(), axis=-1)
            self.labels.update({k: e + 1 + max(self.labels.values()) for e, k in enumerate(default_labels)})
            return self[..., chns]

    def X(self: "Frame2D"):          return self._default(self.CHN.X, self._XY, self.CHN.XY)
    def Y(self: "Frame2D"):          return self._default(self.CHN.Y, self._XY, self.CHN.XY)
    def XY(self: "Frame2D"):         return self._default(self.CHN.XY, self._XY, self.CHN.XY)
    def Z(self: "Frame2D"):          return self[..., self.CHN.Z]
    def RED(self: "Frame2D"):        return self[..., self.CHN.RED]
    def GREEN(self: "Frame2D"):      return self[..., self.CHN.GREEN]
    def BLUE(self: "Frame2D"):       return self[..., self.CHN.BLUE]
    def NB_RED(self: "Frame2D"):     return self[..., self.CHN.NB_RED]
    def NB_GREEN(self: "Frame2D"):   return self[..., self.CHN.NB_GREEN]
    def NB_BLUE(self: "Frame2D"):    return self[..., self.CHN.NB_BLUE]
    def RGB(self: "Frame2D"):        return self[..., self.CHN.RGB]
    def RGBRENIR(self: "Frame2D"):   return self[..., self.CHN.RGBRENIR]
    def HUE(self: "Frame2D"):        return self._default(self.CHN.HUE, self._HSV, self.CHN.HSV)
    def SATURATION(self: "Frame2D"): return self._default(self.CHN.SATURATION, self._HSV,  self.CHN.HSV)
    def VALUE(self: "Frame2D"):      return self._default(self.CHN.VALUE, self._HSV,  self.CHN.VALUE)
    def NDI(self: "Frame2D"):        return self._default(self.CHN.NDI, self._NDI, self.CHN.NDI)
    def EX_G(self: "Frame2D"):       return self._default(self.CHN.EX_G, self._EX_G, self.CHN.EX_G)
    def MEX_G(self: "Frame2D"):      return self._default(self.CHN.MEX_G, self._MEX_G, self.CHN.MEX_G)
    def EX_GR(self: "Frame2D"):      return self._default(self.CHN.EX_GR, self._EX_GR, self.CHN.EX_GR)
    def VEG(self: "Frame2D"):        return self._default(self.CHN.VEG, self._VEG, self.CHN.VEG)

    def _HSV(self: 'Frame2D') -> np.ndarray:
        """ Creates a HSV """
        return rgb2hsv(self.RGB().data)

    def _EX_G(self: 'Frame2D') -> np.ndarray:
        """ Calculates the excessive green index

        2g - 1r - 1b
        """

        return 2 * self.RED().data - \
               self.GREEN().data - \
               self.BLUE().data

    def _MEX_G(self: 'Frame2D') -> np.ndarray:
        """ Calculates the Modified excessive green index

        1.262g - 0.884r - 0.331b"""

        return 1.262 * self.RED().data - \
               0.884 * self.GREEN().data - \
               0.331 * self.BLUE().data

    def _EX_GR(self: 'Frame2D') -> np.ndarray:
        """ Calculates the excessive green minus excess red index

        2g - r - b - 1.4r + g = 3g - 2.4r - b

        :return: np.ndarray, similar shape to callee. Use _get_chns to get as Frame2D
        """

        return 3   * self.RED().data - \
               2.4 * self.GREEN().data - \
                     self.BLUE().data

    def _NDI(self: 'Frame2D') -> np.ndarray:
        """ Calculates the Normalized Difference Index

        (g - r) / (g + r)

        :return: np.ndarray, similar shape to callee. Use _get_chns to get as Frame2D
        """

        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.nan_to_num(
                np.true_divide(self.GREEN().data.astype(np.int) -
                               self.RED  ().data.astype(np.int),
                               self.GREEN().data.astype(np.int) +
                               self.RED  ().data.astype(np.int)),
                copy=False, nan=0, neginf=0, posinf=0)

        return x

    def _VEG(self: 'Frame2D', const_a: float = 0.667) -> np.ndarray:
        """ Calculates the Vegetative Index

        g / {(r^a) * [b^(1 - a)]}

        :param const_a: Constant A depends on the camera used.
        :return: np.ndarray, similar shape to callee. Use _get_chns to get as Frame2D
        """

        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.nan_to_num(self.GREEN().data.astype(np.float) /
                              (np.power(self.RED().data.astype(np.float), const_a) *
                               np.power(self.BLUE().data.astype(np.float), 1 - const_a)),
                              copy=False, nan=0, neginf=0, posinf=0)
        return x

    def _XY(self: 'Frame2D') -> np.ndarray:
        """ Creates the XY Coord Array

        :return: np.ndarray, similar shape to callee. Use _get_chns to get as Frame2D
        """

        # We create a new array to copy self over we expand the last axis by 2
        buffer = np.zeros([*self.data.shape[0:-1], 2])

        # Create X & Y then copy over
        buffer[..., 0] = np.tile(np.arange(0, self.width()), (self.height(), 1))
        buffer[..., 1] = np.tile(np.arange(0, self.height()), (self.width(), 1)).swapaxes(0, 1)

        return buffer
