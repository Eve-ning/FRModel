##################
Available Channels
##################

Here's the glossary of Channels retrievable.

The string version is exactly the name itself. ``f.CHN.RED == "RED"``

In doubt, use the full ``f.CHN.XX`` call as autocomplete will help you.

=================
Non-GLCM Channels
=================

Related :doc:`Examples on retrieving channels <getting_started>`

The syntax is ``f.CHN.XX``.

+--------------+------------------------------------------------------+--------+
| CHN Constant | Description                                          | Layers |
+==============+======================================================+========+
| RED          | Red Channel                                          | 1      |
+--------------+------------------------------------------------------+--------+
| GREEN        | Green Channel                                        | 1      |
+--------------+------------------------------------------------------+--------+
| BLUE         | Blue Channel                                         | 1      |
+--------------+------------------------------------------------------+--------+
| RGB          | A Tuple of Red, Green, Blue Channels                 | 3      |
+--------------+------------------------------------------------------+--------+
| RED_EDGE     | Red Edge Channel                                     | 1      |
+--------------+------------------------------------------------------+--------+
| NIR          | Near Infrared Channel                                | 1      |
+--------------+------------------------------------------------------+--------+
| RGBRENIR     | A Tuple of Red, Green, Blue, Red Edge, NIR Channels  | 5      |
+--------------+------------------------------------------------------+--------+
| X            | X Position                                           | 1      |
+--------------+------------------------------------------------------+--------+
| Y            | Y Position                                           | 1      |
+--------------+------------------------------------------------------+--------+
| XY           | A Tuple of X, Y Channels                             | 2      |
+--------------+------------------------------------------------------+--------+
| HUE          | Hue Channel                                          | 1      |
+--------------+------------------------------------------------------+--------+
| SATURATION   | Saturation Channel                                   | 1      |
+--------------+------------------------------------------------------+--------+
| VALUE        | Value Channel                                        | 1      |
+--------------+------------------------------------------------------+--------+
| HSV          | A Tuple of Hue, Saturation, Value Channels           | 3      |
+--------------+------------------------------------------------------+--------+
| NDI          | Normalized Difference Index Channel                  | 1      |
+--------------+------------------------------------------------------+--------+
| EX_G         | Excess Green Channel                                 | 1      |
+--------------+------------------------------------------------------+--------+
| MEX_G        | Modified Excess Green Channel                        | 1      |
+--------------+------------------------------------------------------+--------+
| EX_GR        | Excess Green Minus Red Channel                       | 1      |
+--------------+------------------------------------------------------+--------+
| VEG          | Vegetation Channel                                   | 1      |
+--------------+------------------------------------------------------+--------+
| NDVI         | Normalized Difference Vegetation Index Channel       | 1      |
+--------------+------------------------------------------------------+--------+
| BNDVI        | Blue Normalized Difference Vegetation Index Channel  | 1      |
+--------------+------------------------------------------------------+--------+
| GNDVI        | Green Normalized Difference Vegetation Index Channel | 1      |
+--------------+------------------------------------------------------+--------+
| GARO         | Green Atmospherically Resistant Vegetation Index     | 1      |
+--------------+------------------------------------------------------+--------+
| GLI          | Green Leaf Index                                     | 1      |
+--------------+------------------------------------------------------+--------+
| GBNDVI       | Green Blue NDVI                                      | 1      |
+--------------+------------------------------------------------------+--------+
| GRNDVI       | Green Red NDVI                                       | 1      |
+--------------+------------------------------------------------------+--------+
| NDRE         | Normalized Difference Red Edge Channel               | 1      |
+--------------+------------------------------------------------------+--------+
| LCI          | Leaf Chlorophyll Index Channel                       | 1      |
+--------------+------------------------------------------------------+--------+
| MSAVI        | Modified Soil Adjusted Vegetation Index Channel      | 1      |
+--------------+------------------------------------------------------+--------+
| OSAVI        | Optimized Soil Adjusted Vegetation Index Channel     | 1      |
+--------------+------------------------------------------------------+--------+

=============
GLCM Channels
=============

Related :doc:`Examples on retrieving channels <getting_started>`

For the channel label, it'll be in the format ``FEATURE_CHANNEL``.

For example, ``CON_RED``, ``COR_GREEN``.

The syntax to retrieve GLCM Feature of CHN is ``f.FEATURE(List[CHN])``.

For example ``f.ASM([f.CHN.RED, f.CHN.GREEN])``.

+--------------+-----------------------+
| GLCM Feature | Description           |
+==============+=======================+
| CON          | Contrast              |
+--------------+-----------------------+
| ASM          | Angular Second Moment |
+--------------+-----------------------+
| COR          | Correlation           |
+--------------+-----------------------+
| MEAN         | GLCM Mean             |
+--------------+-----------------------+
| VAR          | GLCM Variance         |
+--------------+-----------------------+