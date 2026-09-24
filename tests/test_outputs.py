"""What a composite declares about the layers it writes."""

import numpy as np
import pytest

from hls_composites.bands import (
    FMASK,
    NO_PREDICTOR,
    REFLECTANCE_BANDS,
    WIDE_RANGE_PREDICTOR,
)
from hls_composites.indices import DEFAULT_INDICES, NDVI
from hls_composites.outputs import (
    DOY,
    SELECTION_BANDS,
    VALID_COUNT,
    IndexBand,
    OutputBand,
    ReflectanceBand,
    fill_attributes,
    output_bands,
)


class TestOutputBands:
    def test_each_value_band_is_followed_by_its_deviation(self):
        names = [band.name for band in output_bands(DEFAULT_INDICES)]

        assert names == [
            "EVI",
            "EVI_std",
            "NBR",
            "NBR_std",
            "NDVI",
            "NDVI_std",
            "Fmask",
            "ValidCount",
            "DOY",
        ]

    def test_the_selection_layers_are_written_whatever_is_composited(self):
        for values in (DEFAULT_INDICES, REFLECTANCE_BANDS):
            assert [band.name for band in output_bands(values)][-3:] == [
                "Fmask",
                "ValidCount",
                "DOY",
            ]

    def test_bands_output_carries_the_reflectance_bands(self):
        names = [band.name for band in output_bands(REFLECTANCE_BANDS)]

        assert names[:4] == ["red", "red_std", "green", "green_std"]

    def test_every_layer_satisfies_the_protocol(self):
        assert all(
            isinstance(band, OutputBand) for band in output_bands(DEFAULT_INDICES)
        )


class TestDeclarations:
    def test_an_index_layer_takes_its_encoding_from_the_index(self):
        band = IndexBand(NDVI())

        assert band.nodata == NDVI().fill_value
        assert band.scale == NDVI().scale_factor
        assert band.dtype == np.int16
        assert band.long_name == NDVI().long_name

    def test_a_deviation_names_itself_after_its_value_band(self):
        band = IndexBand(NDVI(), deviation=True)

        assert band.name == "NDVI_std"
        assert band.long_name.endswith("standard deviation")
        assert band.nodata == NDVI().fill_value

    def test_a_reflectance_layer_takes_its_encoding_from_the_band_spec(self):
        spec = REFLECTANCE_BANDS[0]
        band = ReflectanceBand(spec)

        assert (band.name, band.nodata, band.scale) == (
            spec.name,
            spec.nodata,
            spec.scale,
        )

    def test_the_selection_layers_are_in_their_own_units(self):
        assert VALID_COUNT.scale == 1.0
        assert DOY.scale == 1.0

    def test_counts_and_days_cannot_be_confused_with_their_fill(self):
        assert VALID_COUNT.nodata < 0
        assert DOY.nodata < 1


class TestPredictors:
    """Differencing pays off on the wide-ranging value bands only."""

    def test_a_value_band_takes_the_predictor_it_declares(self):
        assert IndexBand(NDVI()).predictor == WIDE_RANGE_PREDICTOR
        assert ReflectanceBand(REFLECTANCE_BANDS[0]).predictor == WIDE_RANGE_PREDICTOR

    def test_a_deviation_never_differences(self):
        assert IndexBand(NDVI(), deviation=True).predictor == NO_PREDICTOR
        assert (
            ReflectanceBand(REFLECTANCE_BANDS[0], deviation=True).predictor
            == NO_PREDICTOR
        )

    def test_the_selection_layers_never_difference(self):
        predictors = {band.name: band.predictor for band in SELECTION_BANDS}

        assert predictors == {
            "Fmask": NO_PREDICTOR,
            "ValidCount": NO_PREDICTOR,
            "DOY": NO_PREDICTOR,
        }


class TestFillAttributes:
    def test_value_bands_and_deviations_share_one_attribute(self):
        attributes = fill_attributes({"NDVI": -19999.0, "NDVI_std": -19999.0})

        assert attributes == {"FILLVALUE": -19999}

    def test_each_selection_layer_names_its_own(self):
        attributes = fill_attributes(
            {"Fmask": float(FMASK.nodata), "ValidCount": -999.0, "DOY": -1.0}
        )

        assert attributes == {
            "QA_FILLVALUE": FMASK.nodata,
            "VALIDCOUNT_FILLVALUE": -999,
            "DOY_FILLVALUE": -1,
        }

    def test_a_band_without_a_fill_is_skipped(self):
        assert fill_attributes({"NDVI": None}) == {}

    def test_an_unknown_variable_is_skipped(self):
        assert fill_attributes({"Something": 1.0}) == {}

    def test_disagreeing_fills_under_one_attribute_are_an_error(self):
        """One attribute cannot describe two different fills."""
        with pytest.raises(ValueError, match="FILLVALUE already declares"):
            fill_attributes({"EVI": -19999.0, "NDVI": -1.0})
