import pytest

from unittest.mock import patch, MagicMock
from pathlib import Path

from auto_tune_weights_pipeline.constants import (
    DICTIONARY_PROJECT_NAME_RECOMMENDER_UCP_VIDEO_AND_CLIPS,
)


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.joinpath("data")


@pytest.fixture
def mock_dictionary_hub():
    _patch = "auto_tune_weights_pipeline.features_pairs_generator.requests.post"
    with patch(_patch) as mock_post:
        mock_response = MagicMock()

        _features = {
            "item.counters.vk.comment_v2_vs_card_view.days_7": 14836,
            "item.counters.vk.share_v2_comment_like_vs_like_dislike_share_comment.wilson99": 14823,
            "vk_video.hybrid.eals.60.dot": 8090,
        }

        mock_response.json.return_value = {
            "name2idMap": {
                DICTIONARY_PROJECT_NAME_RECOMMENDER_UCP_VIDEO_AND_CLIPS: {
                    "features": _features,
                }
            }
        }
        mock_post.return_value = mock_response

        yield mock_post
