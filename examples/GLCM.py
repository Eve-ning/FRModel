from frmodel.base.D2 import Frame2D

f = Frame2D.from_image("sample.jpg", scale=0.2)
g = f.get_glcm()
fpl = g.plot()
fpl.image().savefig("sample_out.jpg")