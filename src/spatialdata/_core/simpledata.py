from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import dask.dataframe as dd
import pandas as pd
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame

from spatialdata._types import Raster_T
from spatialdata.models import PointsModel, ShapesModel
from spatialdata.transformations.operations import get_transformation


@dataclass
class SimpleDataLayers:
    images: dict[str, Raster_T] | None = None
    labels: dict[str, Raster_T] | None = None

    def copy(self) -> SimpleDataLayers:
        return SimpleDataLayers(
            images=dict(self.images) if self.images is not None else None,
            labels=dict(self.labels) if self.labels is not None else None,
        )


@dataclass
class SimpleData:
    table: AnnData
    shapes: GeoDataFrame | None = None
    points: DaskDataFrame | None = None
    coordinate_system: str = "global"
    attached: SimpleDataLayers | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        expected_index = pd.Index(self.table.obs_names)
        if not expected_index.is_unique:
            raise ValueError("`table.obs_names` must be unique.")

        if self.shapes is not None:
            ShapesModel.validate(self.shapes)
            self._validate_coordinate_system(self.shapes, "shapes")
            self._validate_exact_index(
                pd.Index(self.shapes.index),
                expected_index,
                "shapes.index",
            )

        if self.points is not None:
            PointsModel.validate(self.points)
            self._validate_coordinate_system(self.points, "points")
            points_index = pd.Index(self.points.compute().index)
            self._validate_exact_index(
                points_index,
                expected_index,
                "points.index",
            )

        if self.attached is not None:
            for name, image in (self.attached.images or {}).items():
                self._validate_coordinate_system(
                    image,
                    f"attached.images[{name!r}]",
                )
            for name, labels in (self.attached.labels or {}).items():
                self._validate_coordinate_system(
                    labels,
                    f"attached.labels[{name!r}]",
                )

    def subset_obs(
        self,
        obs_ids: pd.Index | list[str],
        *,
        copy: bool = True,
    ) -> SimpleData:
        obs_index = pd.Index(obs_ids)
        if not obs_index.is_unique:
            raise ValueError("`obs_ids` must be unique.")

        missing = obs_index.difference(self.table.obs_names)
        if len(missing) > 0:
            raise KeyError(f"Unknown obs ids: {missing.tolist()}")

        table = self.table[obs_index].copy() if copy else self.table[obs_index]

        shapes = None
        if self.shapes is not None:
            subset_shapes = self.shapes.loc[obs_index]
            shapes = subset_shapes.copy() if copy else subset_shapes

        points = None
        if self.points is not None:
            subset_points = self.points.compute().loc[obs_index]
            points = self._rebuild_points_element(subset_points)

        attached = (
            self.attached.copy()
            if copy and self.attached is not None
            else self.attached
        )
        return SimpleData(
            table=table,
            shapes=shapes,
            points=points,
            coordinate_system=self.coordinate_system,
            attached=attached,
        )

    def _rebuild_points_element(self, subset_points: pd.DataFrame) -> DaskDataFrame:
        npartitions = (
            1
            if self.points is None
            else max(1, min(self.points.npartitions, len(subset_points)))
        )
        rebuilt = dd.from_pandas(subset_points, npartitions=npartitions, sort=False)
        rebuilt._attrs = {}
        if self.points is not None:
            rebuilt._attrs.update(deepcopy(self.points.attrs.copy()))
        PointsModel.validate(rebuilt)
        return rebuilt

    def _validate_coordinate_system(self, element: Any, element_name: str) -> None:
        transformations = get_transformation(element, get_all=True)
        if self.coordinate_system not in transformations:
            raise ValueError(
                f"{element_name} does not contain a transformation "
                f"to coordinate system {self.coordinate_system!r}."
            )

    def _validate_exact_index(
        self,
        actual: pd.Index,
        expected: pd.Index,
        element_name: str,
    ) -> None:
        if not actual.is_unique:
            raise ValueError(f"`{element_name}` must be unique.")
        if list(actual) != list(expected):
            missing = expected.difference(actual).tolist()
            extra = actual.difference(expected).tolist()
            raise ValueError(
                f"`{element_name}` must exactly match `table.obs_names`. "
                f"Missing: {missing}; extra: {extra}."
            )
