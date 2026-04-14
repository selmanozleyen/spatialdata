from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from shapely.geometry import Polygon

from spatialdata import SimpleData, SimpleDataLayers
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    ShapesModel,
)


def _make_simpledata() -> SimpleData:
    obs_names = pd.Index(["cell_0", "cell_1", "cell_2"])

    table = AnnData(shape=(3, 0))
    table.obs_names = obs_names

    shapes = gpd.GeoDataFrame(
        {
            "geometry": [
                Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
                Polygon(((1, 0), (1, 1), (2, 1), (2, 0))),
                Polygon(((2, 0), (2, 1), (3, 1), (3, 0))),
            ]
        },
        index=obs_names,
    )
    shapes = ShapesModel.parse(shapes)

    points_df = pd.DataFrame(
        {"x": [0.5, 1.5, 2.5], "y": [0.5, 0.5, 0.5]},
        index=obs_names,
    )
    points = PointsModel.parse(points_df)

    image = Image2DModel.parse(np.zeros((1, 4, 4)), dims=("c", "y", "x"))
    labels = Labels2DModel.parse(np.zeros((4, 4), dtype=int), dims=("y", "x"))

    return SimpleData(
        table=table,
        shapes=shapes,
        points=points,
        attached=SimpleDataLayers(
            images={"image": image},
            labels={"labels": labels},
        ),
    )


def test_simpledata_subset_obs_preserves_alignment_and_order() -> None:
    simple = _make_simpledata()

    subset = simple.subset_obs(["cell_2", "cell_0"])

    assert list(subset.table.obs_names) == ["cell_2", "cell_0"]
    assert list(subset.shapes.index) == ["cell_2", "cell_0"]
    assert list(subset.points.compute().index) == ["cell_2", "cell_0"]
    assert subset.attached is not None
    assert subset.attached is not simple.attached
    assert subset.attached.images is not simple.attached.images
    assert subset.attached.labels is not simple.attached.labels
    assert list(simple.table.obs_names) == ["cell_0", "cell_1", "cell_2"]


def test_simpledata_subset_obs_rejects_unknown_ids() -> None:
    simple = _make_simpledata()

    with pytest.raises(KeyError, match="Unknown obs ids"):
        simple.subset_obs(["cell_0", "missing"])


def test_simpledata_requires_exact_index_alignment() -> None:
    obs_names = pd.Index(["cell_0", "cell_1", "cell_2"])
    table = AnnData(shape=(3, 0))
    table.obs_names = obs_names

    shapes = gpd.GeoDataFrame(
        {
            "geometry": [
                Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
                Polygon(((1, 0), (1, 1), (2, 1), (2, 0))),
                Polygon(((2, 0), (2, 1), (3, 1), (3, 0))),
            ]
        },
        index=pd.Index(["cell_0", "cell_1", "other"]),
    )
    shapes = ShapesModel.parse(shapes)

    with pytest.raises(
        ValueError,
        match="must exactly match `table.obs_names`",
    ):
        SimpleData(table=table, shapes=shapes)
