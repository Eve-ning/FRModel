from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import alphashape
import numpy as np
import utm
from PIL import Image
from PIL import ImageDraw
from scipy.interpolate import CloughTocher2DInterpolator, griddata, LinearNDInterpolator
from scipy.spatial import ConvexHull, Delaunay
from sklearn.impute import KNNImputer

from frmodel.base.D2 import Frame2D

if TYPE_CHECKING:
    from frmodel.base.D3 import Cloud3D

TAG_X_SIZE = 256
TAG_Y_SIZE = 257

class _Cloud3DFrame(ABC):

    @abstractmethod
    def data(self, sample_size=None, transformed=True) -> np.ndarray:
        ...

    @staticmethod
    def _geotiff_to_latlong_ranges(geotiff_path:str) -> tuple:
        im = Image.open(geotiff_path)
        tag = im.tag
        width = tag[TAG_X_SIZE][0]
        height = tag[TAG_Y_SIZE][0]

        # This is the GeoTransform gathered from GDAL
        # Note that the 3rd and 5th are zeros if the tiff is a north-up image (assumption)
        gt = (tag[33922][3], tag[33550][0], 0, tag[33922][4], 0, -tag[33550][1])

        d2latmin, d2longmin = gt[3] + width * gt[4] + height * gt[5], gt[0]
        d2latmax, d2longmax = gt[3], gt[0] + width * gt[1] + height * gt[2]
        return (height, width), (d2latmin, d2latmax), (d2longmin, d2longmax)

    def to_frame(self: 'Cloud3D',
                 geotiff_path: str,
                 samples: int = 100000):
        """ Converts this Cloud3D into a 2D Frame

        This algorithm uses geotiff metadata to fit the Cloud data onto it.

        :param geotiff_path: A Geo-referencable geotiff path
        :param samples: The number of cloud samples to randomly sample for interpolation.
        :return:
        """
        # Extract the UTM Data
        utm_data = np.vstack([self.f.X, self.f.Y, self.f.Z]).astype(np.float)
        utm_min = np.asarray([*self.f.header.min])[..., np.newaxis]
        utm_max = np.asarray([*self.f.header.max])[..., np.newaxis]

        # Get the expected lat long ranges from our GEOTiff
        shape, lat_range, lng_range = self._geotiff_to_latlong_ranges(geotiff_path)

        # For some odd reason, the UTM data isn't scaled correctly to the provided min-max
        # in the header. The incorrect scaled data is prev.
        # Hence, we need to rescale it to its appropriate axis' minmax
        utm_min_prev = np.min(utm_data, axis=1)[..., np.newaxis]
        utm_max_prev = np.max(utm_data, axis=1)[..., np.newaxis]
        utm_data = (utm_data - utm_min_prev) / (utm_max_prev - utm_min_prev) *\
                   (utm_max - utm_min) + utm_min

        # 0: Latitude, 1: lngitude
        utm_data[0], utm_data[1] = utm.to_latlon(utm_data[0], utm_data[1], 48, 'N')

        # The data now is in the correct lat lng.

        # Now, we need to trim out OOB lat lngs
        utm_data = utm_data[:,
                   np.logical_and.reduce((
                       utm_data[0] >= lat_range[0],
                       utm_data[0] <= lat_range[1],
                       utm_data[1] >= lng_range[0],
                       utm_data[1] <= lng_range[1],
                   ))]

        # Finally, we use the provided lat-lng ranges to scale to the shape
        lat_offset = lat_range[0]
        lat_scale = shape[0] / (lat_range[1] - lat_range[0])
        lng_offset = lng_range[0]
        lng_scale = shape[1] / (lng_range[1] - lng_range[0])

        utm_data[0] = (utm_data[0] - lat_offset) * lat_scale
        utm_data[1] = (utm_data[1] - lng_offset) * lng_scale

        rand = np.random.choice(np.arange(0, utm_data.shape[1]), samples if samples else utm_data.shape[1], replace=False)
        x = utm_data[0][rand]
        y = utm_data[1][rand]
        z = utm_data[2][rand]
        X = np.arange(0, shape[0])
        Y = np.arange(0, shape[1])
        XM, YM = np.meshgrid(X, Y)  # 2D grid for interpolation

        # This was supposed to use alphashape but it's very not useful
        # Takes look long and idk why it returns multiple polygons
        # pts = np.stack([x, y], axis=-1)
        # hull = alphashape.alphashape(pts[:samples//10], 0.01)
        # hull_pts = hull.exterior.coords.xy
        #
        # img = Image.new('L', (shape[1], shape[0]), 0)
        # ImageDraw.Draw(img).polygon(np.stack([hull_pts[0], hull_pts[1]], axis=-1),
        #                             outline=1, fill=1)
        # mask = np.asarray(img)

        # interp_z = CloughTocher2DInterpolator(list(zip(x, y)), z, rescale=True)
        interp_z = LinearNDInterpolator(list(zip(x, y)), z, rescale=True)
        Z = interp_z(XM, YM)
        Z = np.where(Z < 0, 0, Z)
        Z = np.nan_to_num(Z)

        # Not sure why the Y is inverted, something to do with the lat long
        f = Frame2D(Z[:,::-1].T[..., np.newaxis], labels=[Frame2D.CHN.Z])
        return f
