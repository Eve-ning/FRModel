from __future__ import annotations

from abc import ABC
from typing import List, TYPE_CHECKING, Union, Iterable, Tuple

import numpy as np

from frmodel.base import CONSTS
from frmodel.base.D2.frame.glcm2 import CyGLCM

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D


class _Frame2DChannelFastGLCM(ABC):
    """ This re-implements Wang Jifei's Fast GLCM Script by adding the option of binarization. """

    def get_glcm(self: 'Frame2D',
                 chns: Iterable[Frame2D.CHN] = (),
                 radius: int = 2,
                 bins: int = 8,
                 step_size: int = 1,
                 pairs: Tuple[str] = ('N', 'W', 'NW', 'SW'),
                 scale_on_bands: bool = True):
        """ This will get the GLCM statistics for this window

        Details on how GLCM works is shown on the wiki.

        :param chns: Channels, can be also in strings.
        :param radius: Radius of GLCM Window
        :param bins: Bin size pre-processing of GLCM.
        :param step_size: The step size of the window.
        """

        assert radius >= 1,    f"Radius should be {radius} >= 1"
        assert bins > 1,      f"Bins should be {bins} > 1"
        assert step_size >= 1, f"Step Size should be {step_size} > 1"

        # FAST GLCM
        chns = chns if chns else list(self.labels.keys())

        self._data = self.data.astype(np.double)

        if scale_on_bands:
            self._data = self.scale_values_on_band().data

        mask = None
        if np.isnan(self.data).any():
            mask = self.nanmask()
            # The np.asarray cast is to remove masking
            # self._data = np.nan_to_num(np.asarray(self.data))

        try:
            data = CyGLCM(self[..., chns].data,
                          radius=radius,
                          bins=bins,
                          step_size=step_size,
                          pairs=pairs).create_glcm()

            data = data.swapaxes(-2, -1).reshape([*data.shape[:2], -1])
        except ValueError:
            return None

        if mask is not None:
            trim = radius + step_size
            data[mask[trim:-trim, trim:-trim]] = np.nan

        labels = []

        labels.extend(CONSTS.CHN.GLCM.HMG( list(self._util_flatten(chns))))
        labels.extend(CONSTS.CHN.GLCM.COR( list(self._util_flatten(chns))))
        labels.extend(CONSTS.CHN.GLCM.ASM( list(self._util_flatten(chns))))
        labels.extend(CONSTS.CHN.GLCM.MEAN(list(self._util_flatten(chns))))
        labels.extend(CONSTS.CHN.GLCM.VAR( list(self._util_flatten(chns))))

        self._data = self.crop_glcm(radius, glcm_by=step_size).data
        t = self.append(data, labels=labels)
        self._data = t.data
        self._labels = t.labels

        return self

    def CON(self, chns: Union[List[str], str]):
        return self[:, :, [f"CON_{i}" for i in chns] if isinstance(chns, List) else f"CON_{chns}"]

    def HMG(self, chns: Union[List[str], str]):
        return self[:, :, [f"HMG_{i}" for i in chns] if isinstance(chns, List) else f"HMG_{chns}"]

    def COR(self, chns: Union[List[str], str]):
        return self[:, :, [f"COR_{i}" for i in chns] if isinstance(chns, List) else f"COR_{chns}"]

    def ASM(self, chns: Union[List[str], str]):
        return self[:, :, [f"ASM_{i}" for i in chns] if isinstance(chns, List) else f"ASM_{chns}"]

    def MEAN(self, chns: Union[List[str], str]):
        return self[:, :, [f"MEAN_{i}" for i in chns] if isinstance(chns, List) else f"MEAN_{chns}"]

    def VAR(self, chns: Union[List[str], str]):
        return self[:, :, [f"VAR_{i}" for i in chns] if isinstance(chns, List) else f"VAR_{chns}"]

