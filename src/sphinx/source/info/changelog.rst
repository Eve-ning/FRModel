#########
ChangeLog
#########

-----
0.0.4
-----

- Separated implementation for ``Frame2D``
- Improved performance for GLCM statistics
- Use Gaussian + FFTConvolution for Non-GLCM Channel fitting
- Implement GLCM statistics with FFTConvolution
- Add Multiprocessing capability for GLCM Entropy
- Split the ``Frame2D`` page into separate implementation pages

-----
0.0.3
-----
- Implement Index Grabbing with ``get_xx`` ops.
- Added shorthand for multiple Index Grabbing with ``get_idxs`` and ``get_all_idxs``.
- **Index** is now a term to represent a generated "channel". That is it's calculated from source data.
- Add from ``from_rgbxy_`` function to enable generation from RGBXY+ arrays. RGB is optional but recommended to include.
- Added GLCM calculation within ``Frame2D``
- Fix issue with ``X`` and ``Y`` consts being flipped
- Added simple wrapper for ``sklearn.neighbours.KDTree`` generation from ``Frame2D``.
- Stage ``Channel2D`` and ``GLCM2D`` for deprecation.
- Force rename **index** to **channel** for differentiation.

-----
0.0.2
-----
- Replace structured array with general ``dtyping`` for efficient coding
- Implement indexes with new data structure

-----
0.0.1
-----
**Initial Commit**
- No changes if there's nothing to change!