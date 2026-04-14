from __future__ import annotations

import geopandas as gpd
import pandas as pd
from anndata import AnnData
from geopandas import GeoDataFrame

from spatialdata import SimpleData, SimpleDataLayers, SpatialData
from spatialdata.models import ShapesModel

def build_cells_simpledata(sdata: SpatialData) -> SimpleData:
    """Build the cell-aligned SimpleData object."""
    table: AnnData = sdata["table"].copy()
    table.obs_names = table.obs["cell_id"].astype(str)
    return SimpleData(
        table=table,
        shapes=sdata["cell_boundaries"].copy(),
        attached=SimpleDataLayers(
            labels={
                "cell_labels": sdata["cell_labels"],
                "nucleus_labels": sdata["nucleus_labels"],
            }
        ),
    )


def build_contours_simpledata(contours_gdf: GeoDataFrame) -> SimpleData:
    """Build a contour-only SimpleData object from a GeoDataFrame."""
    contours_shapes: GeoDataFrame = contours_gdf.set_crs(
        None,
        allow_override=True,
    ).set_index("id").copy()
    contours_shapes = ShapesModel.parse(contours_shapes)

    contours_obs: pd.DataFrame = contours_shapes.drop(columns="geometry").copy()
    contours_obs.index = contours_shapes.index.astype(str)
    contours_table = AnnData(obs=contours_obs)
    return SimpleData(table=contours_table, shapes=contours_shapes)


def select_cells_in_boundaries(
    cells: SimpleData,
    boundaries: SimpleData,
) -> tuple[SimpleData, GeoDataFrame]:
    """Select cells whose centroids lie within boundary polygons."""
    if cells.shapes is None:
        raise ValueError("`cells` must contain shapes.")
    if boundaries.shapes is None:
        raise ValueError("`boundaries` must contain shapes.")

    cell_centroids = gpd.GeoDataFrame(
        geometry=cells.shapes.geometry.centroid,
        index=cells.shapes.index,
    )
    hits: GeoDataFrame = gpd.sjoin(
        cell_centroids,
        boundaries.shapes[["geometry"]],
        how="inner",
        predicate="within",
    )
    selected_ids = hits.index.unique().tolist()
    selected_cells: SimpleData = cells.subset_obs(selected_ids)
    return selected_cells, hits


def select_contours_by_structure(
    contours: SimpleData,
    structure_id: int,
) -> SimpleData:
    """Subset contour SimpleData by structure id."""
    mask = contours.table.obs["structure_id"] == structure_id
    contour_ids = contours.table.obs.index[mask].tolist()
    return contours.subset_obs(contour_ids)
