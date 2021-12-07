import cv2
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from cv2 import findNonZero
from matplotlib import pyplot as plt

from frmodel.base.D2 import Frame2D
from frmodel.streamlit.unsupervised.processing import Processing
from frmodel.streamlit.unsupervised.settings import Settings


def analysis(proc:Processing, settings: Settings):

    trees_len = int(proc.img_wts.max())
    st.metric(f"Number of trees found", trees_len)
    st.caption("Showing significantly sized trees: ")

    # Prepare wts to use with boolean (wts is 1 channel, thus need to have 3 for it to broadcast.)
    img_wts_color = np.repeat(proc.img_wts[..., np.newaxis], proc.img_color.shape[-1], 2)

    img_trees = []

    st.caption("Generating Analysis & GLCM")
    pbar = st.progress(0)
    cols = st.columns(2)
    cols_ix = 0
    for tree_ix in range(1, trees_len):
        tree_bool_3 = img_wts_color == tree_ix
        if np.sum(tree_bool_3) < (settings.img_area * 0.1): continue

        tree_bool = proc.img_wts == tree_ix
        img = np.where(tree_bool_3, proc.img_color, np.nan)

        # This is for RGB
        # Create bins for np.histogram
        bins = np.linspace(0, 255, settings.hist_bins)
        df = pd.DataFrame(
            np.stack(
                [np.histogram(img[..., 0].flatten(), bins=bins)[0],
                 np.histogram(img[..., 1].flatten(), bins=bins)[0],
                 np.histogram(img[..., 2].flatten(), bins=bins)[0]]
            ).T,
            index=bins[:-1],
            columns=['R', 'G', 'B'])

        df = df.reset_index()
        df = df.melt(id_vars=['index'])
        c = alt.Chart(df, height=100).encode(x='index:Q', y='value:Q', color='variable:N').mark_line()
        tree_bool_color = np.where(tree_bool_3, proc.img_color, np.nan)
        cols[cols_ix % 2].image(np.where(tree_bool_3, proc.img_color, proc.img_color // 4), "Tree Image")
        cols[cols_ix % 2].altair_chart(c, True)

        cnts = cv2.findContours(tree_bool.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        x, y, w, h = None, None, None, None

        # This filters for the largest contour bounding rect.
        for cnt in cnts:
            if x is None:
                x, y, w, h = cv2.boundingRect(cnt)
            else:
                x_, y_, w_, h_ = cv2.boundingRect(cnt)
                if (w_ * h_) > (w * h):
                    x, y, w, h = x_, y_, w_, h_

        f_tree = Frame2D(tree_bool_color[y:y+h, x:x+w], Frame2D.CHN.RGB)

        f_tree.get_glcm(radius=settings.glcm_rad, bins=settings.glcm_bins)
        data = f_tree.data
        data = data.reshape([-1, data.shape[-1]])

        bins = np.linspace(0, 1, settings.hist_bins)
        df_hist = pd.DataFrame(np.asarray([np.histogram(data[..., ch], bins)[0] for ch in range(data.shape[-1])]).T,
                         index=bins[:-1],
                         columns=f_tree.channels)
        df_hist = df_hist.reset_index()
        df_hist = df_hist.melt(id_vars=['index'])
        c = alt.Chart(df_hist, height=100).encode(x='index:Q', y='value:Q', color='variable:N')
        cols[cols_ix%2].caption(f"Historgram of GLCM w/ Basebands")
        cols[cols_ix%2].altair_chart(c.mark_line(), True)

        # df = pd.DataFrame(data, columns=f_tree.channels)
        # ar_props = np.asarray([df.mean(), df.var(), df.kurtosis(), df.skew()]).flatten()[..., np.newaxis].T
        # st.image(np.abs(ar_props / ar_props.max()), width=200)

        pbar.progress(int(tree_ix / trees_len * 100))
        cols_ix += 1

    pbar.progress(100)




    cols_wts = st.columns(3)
    for e, (img, c) in enumerate(img_trees):
        cols_wts[e % 3].image(img, width=350)
        cols_wts[e % 3].altair_chart(c)
