from copy import deepcopy
from dataclasses import dataclass

from io import BytesIO


import numpy as np
import requests
import skimage.transform
import streamlit as st
from PIL import Image
from frmodel.base.D2 import Frame2D

@dataclass
class Settings:
    glcm_rad: int
    glcm_bins: int
    glcm_min_size: float
    step_size: int
    hist_bins: int
    rso_size: int
    rsh_size: int
    plm_fp: int
    plm_dist: int
    plm_marker_size: int
    plm_thres_rel: float
    img_area: int
    img_len: int


    f: Frame2D


def settings():
    image_rescale = st.sidebar.slider("Image Rescale Size", 50, 500, 300, step=25)

    with st.spinner("Generating GLCM. Please Wait :)"):
        if st.sidebar.checkbox("Random Cat Image. Courtesy of The Cat API!"):

            url = requests.get('https://api.thecatapi.com/v1/images/search').json()[0]['url']
            response = requests.get(url)
            img_gray = Image.open(BytesIO(response.content))
            ar = skimage.transform.resize(np.asarray(img_gray), [image_rescale, image_rescale, 3])
            st.sidebar.image(ar, width=250)
            f = Frame2D(ar, Frame2D.CHN.RGB)

        else:
            @st.cache
            def read_file_uploader(img):
                if img:
                    read = Image.open(BytesIO(img.read()))
                    ar = np.asarray(read)
                    if ar.ndim == 2:
                        ar = ar[..., np.newaxis]

                    ar = ar[..., :3]
                    ar = skimage.transform.resize(ar,
                                                  [image_rescale, int(image_rescale * ar.shape[1] / ar.shape[0]), ar.shape[-1]])
                    f = Frame2D(ar, [f"CH{c}" for c in range(ar.shape[-1])])

                    return f
                return None


            img_gray = st.sidebar.file_uploader("Upload Image here", [".jpg", ".tif"])
            f = deepcopy(read_file_uploader(img_gray))
            if f is None:
                st.stop()

            f._data = f.data / np.nanmax(f.data)

    glcm_rad = st.sidebar.slider("Radius of GLCM", 1, 25, 1)
    glcm_bins = st.sidebar.slider("Bins of GLCM", 2, 128, 32)
    glcm_min_size = st.sidebar.slider("Min Size to analyze GLCM 0.1%", 1, 100, 10) / 1000
    step_size = st.sidebar.slider("Step Size of GLCM", 1, 10, 3)
    hist_bins = st.sidebar.slider("Histogram Bins", 10, 250, 250)

    img_area = f.shape[0] * f.shape[1]
    img_len = (f.shape[0] + f.shape[1]) / 2
    st.sidebar.subheader("Remove Small Filter")
    rso_size = int(st.sidebar.slider("Objects Size %", 0, 25, 3) * img_area / 10000)
    rsh_size = int(st.sidebar.slider("Holes Size %", 0, 25, 3) * img_area / 10000)

    st.sidebar.subheader("Peak Local Max")
    plm_fp = int(st.sidebar.slider("Footprint %", 0, 25, 1) * img_len / 100)
    plm_dist = int(st.sidebar.slider("Minimum Distance %", 0, 25, 4) * img_len / 100)
    plm_marker_size = st.sidebar.slider("Marker Size", 1, 10, 3)
    plm_thres_rel = st.sidebar.slider("Threshold %", 0, 100, 35) / 100

    return Settings(
        glcm_rad=glcm_rad,
        glcm_bins=glcm_bins,
        glcm_min_size=glcm_min_size,
        step_size=step_size,
        hist_bins=hist_bins,
        rso_size=rso_size,
        rsh_size=rsh_size,
        plm_fp=plm_fp,
        plm_dist=plm_dist,
        plm_marker_size=plm_marker_size,
        plm_thres_rel=plm_thres_rel,
        img_area=img_area,
        img_len=img_len,
        f=f
    )
