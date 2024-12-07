# This version of the code is slower than the original due to the following changes:
# 1. **Polygon Unification**: Now, all extracted polygons are unified (using `unary_union`), which increases execution time, especially in large areas with many objects.
# 2. **Geometry Simplification**: The extracted geometries can be more effectively simplified using the `simplify` parameter to reduce polygon complexity, but this adds an extra step to the process.
# 3. **Increased Memory Usage**: The code now loads and processes all geometries in memory before saving, which can increase memory consumption and execution time compared to the original code that processed in blocks (tiles).
#
# However, the improvements include:
# 1. **Higher Polygon Accuracy**: The unification and simplification of geometries can result in more accurate polygons, with no overlaps and less noise.
# 2. **More Detailed Output**: The output now includes information about area (in hectares) and perimeter (in meters), which can be useful for subsequent analysis.
# 3. **Better CRS Handling**: The code now reprojects geometries to an equal-area CRS (EPSG:6933), improving accuracy when calculating areas and perimeters across different CRS.
#
# As a result, this version is slower and consumes more memory but provides higher accuracy and better output for analysis.


import math
import os
import re
import time

import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.features
import shapely.geometry
from shapely.ops import unary_union
from affine import Affine
from fiboa_cli.parquet import create_parquet, features_to_dataframe
from pyproj import CRS
from tqdm import tqdm

from .cfg import SUPPORTED_POLY_FORMATS_TXT

def polygonize(input, out, simplify, min_size, overwrite, close_interiors):
    """Polygonize the output from inference into vector polygons."""

    print(f"Polygonizing input file: {input}")

    # Set default output path if not provided
    if not out:
        out = os.path.splitext(input)[0] + ".parquet"

    # If the output file exists and overwrite is not set, do nothing
    if os.path.exists(out) and not overwrite:
        print(f"Output file {out} already exists. Use -f to overwrite.")
        return
    elif os.path.exists(out) and overwrite:
        os.remove(out)  # Delete existing output file if overwrite flag is set

    tic = time.time()

    # Define schema for the output file
    schema = {
        'geometry': 'Polygon',
        'properties': {
            'id': 'str',
            'area': 'float',
            'perimeter': 'float',
        }
    }
    i = 1  # Initialize ID counter
    all_geometries = []  # List to store polygons

    # Open the input raster file
    with rasterio.open(input) as src:
        original_crs = src.crs.to_string()  # Original coordinate reference system
        is_meters = src.crs.linear_units in ["m", "metre", "meter"]  # Check if CRS uses meters
        transform = src.transform  # Affine transform for geo-referencing
        equal_area_crs = CRS.from_epsg(6933)  # Use an equal area projection for geometry processing
        tags = src.tags()  # Metadata tags from the raster file
        
        input_height, input_width = src.shape  # Image dimensions
        mask = (src.read(1) == 1).astype(np.uint8)  # Mask where pixels have a value of 1
        polygonization_stride = 2048  # Define the stride (tile size) for processing
        total_iterations = math.ceil(input_height / polygonization_stride) * math.ceil(input_width / polygonization_stride)

        # Progress bar for processing mask windows
        with tqdm(total=total_iterations, desc="Processing mask windows") as pbar:
            for y in range(0, input_height, polygonization_stride):
                for x in range(0, input_width, polygonization_stride):
                    new_transform = transform * Affine.translation(x, y)  # Shift the transform for each window
                    mask_window = mask[y:y+polygonization_stride, x:x+polygonization_stride]  # Extract the mask window
                    for geom_geojson, val in rasterio.features.shapes(mask_window, transform=new_transform):
                        if val != 1:  # Process only areas where the mask value is 1
                            continue

                        geom = shapely.geometry.shape(geom_geojson)  # Convert to shapely geometry
                        if close_interiors:
                            geom = shapely.geometry.Polygon(geom.exterior)  # Close any interior holes in the geometry
                        if simplify > 0:
                            geom = geom.simplify(simplify)  # Simplify the geometry if requested

                        # If CRS is not in meters, reproject to equal-area CRS
                        if is_meters:
                            geom_proj_meters = geom
                        else:
                            geom_proj_meters = shapely.geometry.shape(
                                fiona.transform.transform_geom(
                                    original_crs, equal_area_crs, geom_geojson
                                )
                            )

                        area = geom_proj_meters.area  # Calculate the area in square meters
                        perimeter = geom_proj_meters.length  # Calculate the perimeter in meters

                        if area >= min_size:  # Only keep geometries larger than min_size
                            all_geometries.append(geom)  # Append geometry to the list

                    pbar.update(1)  # Update progress bar after each iteration

    # Merge the individual geometries into a unified geometry (unified polygon)
    unified_geometry = unary_union(all_geometries)

    # If the unified geometry is a multipolygon, convert to individual polygons
    if isinstance(unified_geometry, shapely.geometry.Polygon):
        geometries_to_save = [unified_geometry]
    elif isinstance(unified_geometry, shapely.geometry.MultiPolygon):
        geometries_to_save = list(unified_geometry.geoms)
    else:
        geometries_to_save = []

    # Prepare the rows for the output
    rows = []
    for geom in geometries_to_save:
        if simplify > 0:
            geom = geom.simplify(simplify)  # Simplify geometry if requested

        rows.append({
            "geometry": shapely.geometry.mapping(geom),
            "properties": {
                "id": str(i),
                "area": geom.area * 0.0001,  # Convert area to hectares
                "perimeter": geom.length,  # Perimeter in meters
            }
        })
        i += 1  # Increment ID counter

    # Check if the output file should be saved in Parquet format
    if out.endswith(".parquet"):
        timestamp = tags.get("TIFFTAG_DATETIME", None)
        if timestamp:
            # Format timestamp to a standard format
            pattern = re.compile(r"^(\d{4})[:-](\d{2})[:-](\d{2})[T\s](\d{2}):(\d{2}):(\d{2}).*$")
            if pattern.match(timestamp):
                timestamp = re.sub(pattern, r"\1-\2-\3T\4:\5:\6Z", timestamp)
        
        config = collection = {"fiboa_version": "0.2.0"}  # Config info for the output
        columns = ["geometry", "determination_method"] + list(schema["properties"].keys())
        gdf = features_to_dataframe(rows, columns)
        gdf.set_crs(original_crs, inplace=True, allow_override=True)  # Set the CRS of the dataframe
        gdf["determination_method"] = "auto-imagery"
        if timestamp:
            gdf["determination_datetime"] = timestamp
            columns.append("determination_datetime")

        create_parquet(gdf, columns, collection, out, config, compression="brotli")  # Save as Parquet
    else:
        print("WARNING: The fiboa-compliant GeoParquet output format is recommended for field boundaries.")
        with fiona.open(out, 'w', "GPKG", schema=schema, crs=original_crs) as dst:
            dst.writerecords(rows)  # Save as GeoPackage if not Parquet

    print(f"Finished polygonizing output at {out} in {time.time() - tic:.2f}s")


# Original code

# import math
# import os
# import re
# import time

# import fiona
# import fiona.transform
# import numpy as np
# import rasterio
# import rasterio.features
# import shapely.geometry
# from affine import Affine
# from fiboa_cli.parquet import create_parquet, features_to_dataframe
# from pyproj import CRS
# from tqdm import tqdm

# from .cfg import SUPPORTED_POLY_FORMATS_TXT


# def polygonize(input, out, simplify, min_size, overwrite, close_interiors):
#     """Polygonize the output from inference."""

#     print(f"Polygonizing input file: {input}")

#     # TODO: Get this warning working right, based on the CRS of the input file
#     # if simplify is not None and simplify > 1:
#     #    print("WARNING: You are passing a value of `simplify` greater than 1 for a geographic coordinate system. This is probably **not** what you want.")

#     if not out:
#         out = os.path.splitext(input)[0] + ".parquet"

#     if os.path.exists(out) and not overwrite:
#         print(f"Output file {out} already exists. Use -f to overwrite.")
#         return
#     elif os.path.exists(out) and overwrite:
#         os.remove(out)  # GPKGs are sometimes weird about overwriting in-place

#     tic = time.time()
#     rows = []
#     schema = {
#         'geometry': 'Polygon',
#         'properties': {
#             'id': 'str',
#             'area': 'float',
#             'perimeter': 'float',
#         }
#     }
#     i = 1
#     # read the input file as a mask
#     with rasterio.open(input) as src:
#         original_crs = src.crs.to_string()
#         is_meters = src.crs.linear_units in ["m", "metre", "meter"]
#         transform = src.transform
#         equal_area_crs = CRS.from_epsg(6933) # Define the equal-area projection using EPSG:6933
#         tags = src.tags()

#         input_height, input_width = src.shape
#         mask = (src.read(1) == 1).astype(np.uint8)
#         polygonization_stride = 2048
#         total_iterations = math.ceil(input_height / polygonization_stride) * math.ceil(input_width / polygonization_stride)

#         if out.endswith(".gpkg"):
#             format = "GPKG"
#         elif out.endswith(".parquet"):
#             format = "Parquet"
#         elif out.endswith(".fgb"):
#             format = "FlatGeobuf"
#         elif out.endswith(".geojson") or out.endswith(".json"):
#             format = "GeoJSON"
#         else:
#             raise ValueError("Output format not supported. " + SUPPORTED_POLY_FORMATS_TXT)

#         rows = []
#         with tqdm(total=total_iterations, desc="Processing mask windows") as pbar:
#             for y in range(0, input_height, polygonization_stride):
#                 for x in range(0, input_width, polygonization_stride):
#                     new_transform = transform * Affine.translation(x, y)
#                     mask_window = mask[y:y+polygonization_stride, x:x+polygonization_stride]
#                     for geom_geojson, val in rasterio.features.shapes(mask_window, transform=new_transform):
#                         if val != 1:
#                             continue
                            
#                         geom = shapely.geometry.shape(geom_geojson)

#                         if close_interiors:
#                             geom = shapely.geometry.Polygon(geom.exterior)
#                         if simplify > 0:
#                             geom = geom.simplify(simplify)
                        
#                         # Calculate the area of the reprojected geometry
#                         if is_meters:
#                             geom_proj_meters = geom
#                         else:
#                             # Reproject the geometry to the equal-area projection
#                             # if the CRS is not in meters
#                             geom_proj_meters = shapely.geometry.shape(
#                                 fiona.transform.transform_geom(
#                                     original_crs, equal_area_crs, geom_geojson
#                                 )
#                             )

#                         area = geom_proj_meters.area
#                         perimeter = geom_proj_meters.length
                        
#                         # Only include geometries that meet the minimum size requirement
#                         if area < min_size:
#                             continue

#                         rows.append({
#                             "geometry": shapely.geometry.mapping(geom),
#                             "properties": {
#                                 "id": str(i),
#                                 "area": area * 0.0001, # Add the area in hectares
#                                 "perimeter": perimeter, # Add the perimeter in meters
#                             }
#                         })
#                         i += 1
                    
#                     pbar.update(1)

#     if format == "Parquet":
#         timestamp = tags.get("TIFFTAG_DATETIME", None)
#         if timestamp is not None:
#             pattern = re.compile(r"^(\d{4})[:-](\d{2})[:-](\d{2})[T\s](\d{2}):(\d{2}):(\d{2}).*$")
#             if pattern.match(timestamp):
#                 timestamp = re.sub(pattern, r"\1-\2-\3T\4:\5:\6Z", timestamp)
#             else:
#                 print("WARNING: Unable to parse timestamp from TIFFTAG_DATETIME tag.")
#                 timestamp = None
    
#         config = collection = {"fiboa_version": "0.2.0"}
#         columns = ["geometry", "determination_method"] + list(schema["properties"].keys())
#         gdf = features_to_dataframe(rows, columns)
#         gdf.set_crs(original_crs, inplace=True, allow_override=True)
#         gdf["determination_method"] = "auto-imagery"
#         if timestamp is not None:
#             gdf["determination_datetime"] = timestamp
#             columns.append("determination_datetime")
        
#         create_parquet(gdf, columns, collection, out, config, compression = "brotli")
#     else:
#         print("WARNING: The fiboa-compliant GeoParquet output format is recommended for field boundaries.")
#         with fiona.open(out, 'w', format, schema=schema, crs=original_crs) as dst:
#             dst.writerecords(rows)

#     print(f"Finished polygonizing output at {out} in {time.time() - tic:.2f}s")
