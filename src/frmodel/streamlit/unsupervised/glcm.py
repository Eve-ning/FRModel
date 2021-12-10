import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from frmodel.streamlit.unsupervised.processing import Processing
from frmodel.streamlit.unsupervised.settings import Settings


def analysis(proc:Processing, settings: Settings):

    # The number of trees
    trees_len = int(proc.img_wts.max())

    # Show Tree Count
    st.metric(f"Number of trees found", trees_len)


    st.caption("Showing significantly sized trees: ")

    img_wts_color = np.repeat(proc.img_wts[..., np.newaxis], proc.img_color.shape[-1], 2)

    imgs_wts = []

    for tree_ix in range(1, trees_len):
        tree_bool = img_wts_color == tree_ix
        if np.sum(tree_bool) < (settings.img_area * 0.1): continue
        img = np.where(tree_bool, proc.img_color, np.nan)

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
        imgs_wts.append((np.where(tree_bool, proc.img_color, 0), c))

    cols_wts = st.columns(3)
    for e, (img, c) in enumerate(imgs_wts):
        cols_wts[e % 3].image(img, width=350)
        cols_wts[e % 3].altair_chart(c)
