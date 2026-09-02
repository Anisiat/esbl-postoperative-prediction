"""Tests for mapping longitudinal evidence to Charlson categories."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.features.build_comorbidity_features import (
    CCI_WEIGHTS,
    aggregate_comorbidities_by_infection_episode,
    map_comorbidities_to_cci,
    reconcile_cci_mapping,
)


def test_map_comorbidities_uses_prefix_exact_and_multilabel_matching() -> None:
    combined = pd.DataFrame(
        {
            "icd_code": ["E11.9", "N179", pd.NA],
            "snomed_code": ["194781004", "38341003", pd.NA],
            "medication_name": [" Tiotropium ", "aspirin", pd.NA],
        }
    )
    original = combined.copy(deep=True)
    mappings = {
        "icd": pd.DataFrame(
            {
                "comorbidity": ["diabetes_without_complication"],
                "icd_code": ["E11"],
                "weight": [1],
            }
        ),
        "snomed": pd.DataFrame(
            {
                "comorbidity": [
                    "congestive_heart_failure",
                    "renal_disease",
                ],
                "snomed_code": ["194781004", "194781004"],
            }
        ),
        "medication": pd.DataFrame(
            {
                "comorbidity": ["chronic_pulmonary_disease"],
                "medication_name": ["tiotropium"],
            }
        ),
    }

    result = map_comorbidities_to_cci(combined, mappings)

    assert result["icd_comorbidity"].tolist() == [
        ("diabetes_without_complication",),
        (),
        (),
    ]
    assert result["snomed_comorbidity"].tolist() == [
        ("congestive_heart_failure", "renal_disease"),
        (),
        (),
    ]
    assert result["medication_comorbidity"].tolist() == [
        ("chronic_pulmonary_disease",),
        (),
        (),
    ]
    assert_frame_equal(combined, original)


def test_reconcile_cci_mapping_handles_empty_and_multilabel_tuples() -> None:
    mapped = pd.DataFrame(
        {
            "icd_comorbidity": [
                ("congestive_heart_failure",),
                (),
                ("congestive_heart_failure",),
                ("malignancy",),
                (),
                (),
            ],
            "snomed_comorbidity": [
                (),
                ("renal_disease",),
                ("congestive_heart_failure", "renal_disease"),
                ("dementia",),
                (),
                (),
            ],
            "medication_comorbidity": [
                (),
                (),
                (),
                (),
                ("chronic_pulmonary_disease",),
                (),
            ],
        }
    )

    result = reconcile_cci_mapping(mapped)

    assert result["coding_conflict"].tolist() == [0, 0, 0, 1, 0, 0]
    assert result["cci_condition"].tolist() == [
        ("congestive_heart_failure",),
        ("renal_disease",),
        ("congestive_heart_failure", "renal_disease"),
        (),
        ("chronic_pulmonary_disease",),
        (),
    ]


def test_aggregate_comorbidities_builds_time_aware_episode_flags() -> None:
    cci_long = pd.DataFrame(
        {
            "subject": ["P001", "P001", "P001", "P001", "P002"],
            "comorbidity_date": [
                "2024-01-01 12:00:00",
                "2024-02-01",
                "2024-04-01",
                None,
                "2024-01-10",
            ],
            "cci_condition": [
                ("myocardial_infarction",),
                ("congestive_heart_failure", "renal_disease"),
                ("dementia",),
                ("aids_hiv",),
                "chronic_pulmonary_disease",
            ],
        }
    )
    infection_episodes = pd.DataFrame(
        {
            "subject": ["P001", "P001", "P002", "P003"],
            "infection_id": ["INF001", "INF002", "INF003", "INF004"],
            "admission_date": [
                "2024-02-01",
                "2024-05-01",
                "2024-01-09",
                "2024-05-01",
            ],
        }
    )
    original_cci = cci_long.copy(deep=True)
    original_episodes = infection_episodes.copy(deep=True)

    result = aggregate_comorbidities_by_infection_episode(
        cci_long,
        infection_episodes,
    )

    expected = pd.DataFrame(
        0,
        index=pd.MultiIndex.from_tuples(
            [
                ("P001", "INF001"),
                ("P001", "INF002"),
                ("P002", "INF003"),
                ("P003", "INF004"),
            ],
            names=["subject", "infection_id"],
        ),
        columns=list(CCI_WEIGHTS),
        dtype="int64",
    )
    expected.loc[("P001", "INF001"), "myocardial_infarction"] = 1
    expected.loc[("P001", "INF001"), "congestive_heart_failure"] = 1
    expected.loc[("P001", "INF001"), "renal_disease"] = 1
    expected.loc[("P001", "INF002"), "myocardial_infarction"] = 1
    expected.loc[("P001", "INF002"), "congestive_heart_failure"] = 1
    expected.loc[("P001", "INF002"), "renal_disease"] = 1
    expected.loc[("P001", "INF002"), "dementia"] = 1

    assert_frame_equal(result, expected)
    assert set(result.to_numpy().ravel()) <= {0, 1}
    assert_frame_equal(cci_long, original_cci)
    assert_frame_equal(infection_episodes, original_episodes)


def test_aggregate_comorbidities_rejects_duplicate_episode_keys() -> None:
    cci_long = pd.DataFrame(
        columns=["subject", "comorbidity_date", "cci_condition"]
    )
    infection_episodes = pd.DataFrame(
        {
            "subject": ["P001", "P001"],
            "infection_id": ["INF001", "INF001"],
            "admission_date": ["2024-01-01", "2024-01-01"],
        }
    )

    with pytest.raises(ValueError, match="duplicate keys"):
        aggregate_comorbidities_by_infection_episode(
            cci_long,
            infection_episodes,
        )
