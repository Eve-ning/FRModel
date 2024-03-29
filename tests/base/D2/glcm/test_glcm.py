import unittest

import numpy as np

from frmodel.base.D2.frame2D import Frame2D
from frmodel.base.consts import CONSTS
from tests.base.D2Fixture.test_fixture import TestD2Fixture


class TestGLCM(TestD2Fixture):

    def test_glcm(self):
        """ Custom validated and calculated GLCM Test. See Journal for example. """
        ar = np.asarray([[5, 8, 9, 5],
                         [0, 0, 1, 7],
                         [6, 9, 2, 4],
                         [5, 2, 4, 2]]).transpose()[..., np.newaxis]

        f = Frame2D(ar.astype(np.uint8), CONSTS.CHN.RED)
        fc = f.get_chns(self_=False,
                        glcm=Frame2D.GLCM(radius=1,
                                          channels=[f.CHN.RED]))

        # Will need to verify this again manually
        # """ The reason why I made calling this so verbose is to make it easy for development. """
        #
        # self.assertAlmostEqual(fc.data_chn(fc.CHN.GLCM.CON(fc.CHN.RED)).data.squeeze(), 213)
        # self.assertAlmostEqual(fc.data_chn(fc.CHN.GLCM.COR(fc.CHN.RED)).data.squeeze(), -0.12209306360906494,
        #                        places=4)
        # self.assertAlmostEqual(fc.data_chn(fc.CHN.GLCM.ASM(fc.CHN.RED)).data.squeeze(), 1)


if __name__ == '__main__':
    unittest.main()
