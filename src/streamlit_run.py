import streamlit as st
from frmodel.streamlit.unsupervised.analysis import analysis
from frmodel.streamlit.unsupervised.settings import settings
from frmodel.streamlit.unsupervised.processing import processing


st.set_page_config(layout='wide')

"""
# GLCM Implementation
"""
c = st.columns(2)

settings = settings()
proc = processing(settings)
analysis(proc, settings)


with st.expander("Function Definitions"):
    st.latex("\\text{Homogeneity} = \sum_{i,j=0}^{N-1}\\frac{P_{i,j}}{1+(i-j)^2}")
    st.latex("\\text{Angular Second Moment (ASM)} = \sum_{i,j=0}^{N-1}P_{i,j}^2")
    st.latex("\\text{GLCM Mean i, } \mu_i = \sum_{i,j=0}^{N-1}i * P_{i,j}")
    st.latex("\\text{GLCM Mean j, } \mu_j = \sum_{i,j=0}^{N-1}j * P_{i,j}")
    st.latex("\\text{GLCM Mean, } \mu = (\mu_i + \mu_j) / 2")
    st.latex("\\text{GLCM Variance i, } \sigma_i^2 = \sum_{i,j=0}^{N-1}P_{i,j}(i - \mu_i)^2")
    st.latex("\\text{GLCM Variance j, } \sigma_j^2 = \sum_{i,j=0}^{N-1}P_{i,j}(j - \mu_j)^2")
    st.latex("\\text{GLCM Variance, } \sigma^2 = (\sigma_i^2 + \sigma_j^2) / 2")
    st.latex("\\text{Correlation} = \\frac{(i - \mu_i)(j - \mu_j)}{\sqrt{\sigma_i^2\sigma_j^2}}")



# cnts = sorted(cnts, key=lambda x: cv2.arcLength(x, True), reverse=True)
# for e, cnt in enumerate(cnts[:3]):
#     cnt_centre, cnt_rad = cv2.minEnclosingCircle(cnt)
#     cnt_img = cv2.circle(img_color.copy(),
#                          (int(cnt_centre[0]), int(cnt_centre[1])),
#                          int(cnt_rad),
#                          (255, 0, 0), 2)
#     cols[e].image(cnt_img, f"Contour Circle {e}")


#
# cols = st.columns(6)
# for c in range(f.shape[-1]):
#     cols[c%6].subheader(list(f.labels)[c])
#     data = f.data[...,c]
#     cols[c%6].image(data, width=image_display, clamp=True)
#     df = pd.DataFrame(np.histogram(data[~np.isnan(data)],
#                                    bins=np.linspace(0,1,hist_bins))[0],
#                       columns=['hist'],
#                       index=np.linspace(0,1,hist_bins)[:-1])
#     df = df.reset_index()
#     ch = alt.Chart(df, height=200).encode(x='index:Q', y='hist:Q').mark_bar()
#     cols[c%6].altair_chart(ch, True)