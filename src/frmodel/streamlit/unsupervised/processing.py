from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import cv2

import numpy as np
import streamlit as st
from scipy.ndimage import distance_transform_edt
from skimage.draw import circle_perimeter
from skimage.feature import peak_local_max
from skimage.filters import try_all_threshold, threshold_yen
from skimage.measure import label
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import watershed

from frmodel.base.D2 import Frame2D
from frmodel.streamlit.unsupervised.settings import Settings

@dataclass
class Processing:
    img_color:      np.ndarray = None; img_color_txt:      str = None
    img_gray:       np.ndarray = None; img_gray_txt:       str = None
    img_yen:        np.ndarray = None; img_yen_txt:        str = None
    img_rs:         np.ndarray = None; img_rs_txt:         str = None
    img_edt:        np.ndarray = None; img_edt_txt:        str = None
    img_peaks:      np.ndarray = None; img_peaks_txt:      str = None
    img_wts:        np.ndarray = None; img_wts_txt:        str = None
    img_canny_cnts: np.ndarray = None; img_canny_cnts_txt: str = None
    img_canny:      np.ndarray = None; img_canny_txt:      str = None

    @property
    def imgs(self):
        return [
            self.img_color,
            self.img_gray,
            self.img_yen,
            self.img_rs,
            self.img_edt,
            self.img_peaks,
            self.img_wts,
            self.img_canny_cnts,
            self.img_canny,
        ]
    @property
    def img_txts(self):
        return [
            self.img_color_txt,
            self.img_gray_txt,
            self.img_yen_txt,
            self.img_rs_txt,
            self.img_edt_txt,
            self.img_peaks_txt,
            self.img_wts_txt,
            self.img_canny_cnts_txt,
            self.img_canny_txt,
        ]

def processing(settings: Settings):
    p = Processing()

    img_color = (deepcopy(settings.f).data * 255).astype(np.uint8)
    p.img_color, p.img_color_txt = (img_color, "Original Image")

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    p.img_gray, p.img_gray_txt = (img_gray, "Gray Image")

    img_thresh = threshold_yen(img_gray)
    img_yen = img_gray > img_thresh

    p.img_yen, p.img_yen_txt = (img_yen * 255, "Yen Threshold")

    with st.expander("All Thresholds"):
        fig, ax = try_all_threshold(img_gray, figsize=(10, 8), verbose=False)
        st.pyplot(fig)

    img_rs = remove_small_holes(remove_small_objects(img_yen, settings.rso_size), settings.rsh_size)

    p.img_rs, p.img_rs_txt = (img_rs * 255, "Remove Small Holes & Objects")

    img_edt = distance_transform_edt(img_rs)
    p.img_edt, p.img_edt_txt = (img_edt / img_edt.max(), "Euclidean distance transform")

    img_peak_coords = \
        peak_local_max(img_edt,
                       footprint=np.ones([settings.plm_fp, settings.plm_fp]),
                       min_distance=settings.plm_dist,
                       exclude_border=False,
                       threshold_rel=settings.plm_thres_rel)

    # This simply just creates the EDT Image for web display
    # Understanding this is not necessary.
    img_peaks = cv2.cvtColor(((img_edt / img_edt.max() * 255).copy()).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    for e, plm in enumerate(img_peak_coords):
        rr, cc = circle_perimeter(*plm, settings.plm_marker_size)
        for r, c in zip(rr, cc):
            if r < img_peaks.shape[0] and c < img_peaks.shape[1]:
                img_peaks[r, c] = [255, 0, 0]

        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img_peaks, str(e), (plm[1], plm[0]), font, 1, (255, 255, 255), 2)

    p.img_peaks, p.img_peaks_txt = (img_peaks, "Peak Local Maxima")

    img_peaks_mask = np.zeros_like(img_edt, dtype=bool)
    img_peaks_mask[tuple(img_peak_coords.T)] = 1
    img_peaks_markers = label(img_peaks_mask)
    img_wts = watershed(-img_edt, img_peaks_markers, mask=img_rs)
    p.img_wts, p.img_wts_txt = (img_wts, "Watershed")

    img_canny = cv2.Canny(img_wts.astype(np.uint8), 0, 1)
    img_canny_cnts_, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_canny_cnts = cv2.drawContours(np.zeros_like(img_color), img_canny_cnts_, -1, (255, 0, 0), -1)

    p.img_canny_cnts, p.img_canny_cnts_txt = (img_canny_cnts, "Canny Contour Image")
    img_canny = cv2.drawContours(img_color.copy(), img_canny_cnts_, -1, (255, 0, 0), -1)

    p.img_canny, p.img_canny_txt = (img_canny, "Canny Image")

    # Renders the images in a gallery fashion
    cols = st.columns(2)
    for e, (img, txt) in enumerate(zip(p.imgs, p.img_txts)):
        cols[e % 2].image(img, txt, width=600)

    return p