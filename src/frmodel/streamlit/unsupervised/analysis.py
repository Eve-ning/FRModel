import cv2
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from frmodel.base.D2 import Frame2D
from frmodel.streamlit.unsupervised.processing import Processing
from frmodel.streamlit.unsupervised.settings import Settings


def analysis(proc: Processing, settings: Settings):

    # The number of trees
    trees_len = int(proc.img_wts.max())

    # Show Tree Count
    st.metric(f"Number of trees found", trees_len)
    st.caption("Showing significantly sized trees: ")

    # Prepare watershed to use with boolean (wts is 1 channel, thus need to have 3 for it to broadcast.)
    img_wts_3 = np.repeat(proc.img_wts[..., np.newaxis], proc.img_color.shape[-1], 2)

    st.caption("Generating Analysis & GLCM")

    # Progress Bar.
    # This will not be consistent as we don't progress for skipped trees.
    pbar = st.progress(0)

    relative_hist = st.checkbox("Relative Histogram", value=True)
    combine_hist  = st.checkbox("Combined Histogram", value=True)

    # 2 Columns to show trees. The height of each entry should be consistent
    cols = st.columns(2)

    # This is just to swap between the 2 columns for each tree
    cols_ix = 0

    for tree_ix in range(1, trees_len):

        # This simply yields watershed part of specified index
        tree_bool_3 = img_wts_3 == tree_ix

        # Skip GLCM processing if the tree is too small
        if np.sum(tree_bool_3) < (settings.img_area * settings.glcm_min_size): continue

        # This yields the img (with nan) of the watershed boolean ix
        img = np.where(tree_bool_3, proc.img_color, np.nan)

        # This generates the RGB hist analysis pre glcm

        tree_bool_color = np.where(tree_bool_3, proc.img_color, np.nan)
        cols[cols_ix % 2].image(np.where(tree_bool_3, proc.img_color, proc.img_color // 2), "Tree Image")

        # This is just the boolean with only 1 channel
        # This is required for the findContours
        tree_bool_1 = proc.img_wts == tree_ix

        # Find the contour of the 1 channel boolean img and yield the bounding rect
        # This is to retrieve the smallest Frame2D for quick GLCM processing
        cnts = cv2.findContours(tree_bool_1.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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

        # Yield the smallest Frame2D from the bounding rect
        f_tree = Frame2D(tree_bool_color[y:y+h, x:x+w] / 255, [f'Channel {k}' for k in range(img.shape[-1])])

        glcm_success = \
            f_tree.get_glcm(radius=settings.glcm_rad, bins=settings.glcm_bins, step_size=settings.step_size,
                        scale_on_bands=False)
        if glcm_success is None:
            cols[cols_ix % 2].markdown("**GLCM Failed to Generate due to radius of GLCM**")
            continue
        # Analyzes the GLCM
        data = f_tree.data

        data = data.reshape([-1, data.shape[-1]])
        bins = np.linspace(0, 1, settings.hist_bins)

        def hist(df_hist_):
            if relative_hist: df_hist_ /= df_hist_.max()
            df_hist_ = df_hist_.reset_index()
            df_hist_ = df_hist_.melt(id_vars=['index'])
            c = alt.Chart(df_hist_, height=100).encode(x='index:Q', y='value:Q', color='variable:N')
            cols[cols_ix % 2].altair_chart(c.mark_line(), True)

        if combine_hist:
            df_hist = pd.DataFrame(np.asarray([np.histogram(data[..., ch], bins)[0]
                                               for ch in range(data.shape[-1])]).T,
                             index=bins[:-1],
                             columns=f_tree.channels)
            hist(df_hist)
        else:
            for k in range(6):
                channels = data.shape[-1] // 6
                df_hist = pd.DataFrame(np.asarray([np.histogram(data[..., ch], bins)[0]
                                                   for ch in range(k * channels, k * channels + channels)]).T,
                                 index=bins[:-1],
                                 columns=f_tree.channels[k * channels: k * channels + channels])
                hist(df_hist)

        # This is just for the fingerprinting, however, since the values are not bounded, it's kinda hard to get a
        # logical fingerprint

        # df = pd.DataFrame(data, columns=f_tree.channels)
        # ar_props = np.asarray([df.mean(), df.var(), df.kurtosis(), df.skew()]).flatten()[..., np.newaxis].T
        # st.image(np.abs(ar_props / ar_props.max()), width=200)

        pbar.progress(int(tree_ix / trees_len * 100))
        cols_ix += 1

    pbar.progress(100)


