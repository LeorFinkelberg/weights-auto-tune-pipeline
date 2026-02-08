from enum import StrEnum, auto


RANDOM_SEED = 42424242
BIG_NEGATIVE_DEFAULT_VALUE = -3.4e38
DICTIONARY_HUB_URL = "https://dictionary-hub.kaizen.idzn.ru/fetch"
DICTIONARY_PROJECT_NAME_RECOMMENDER_UCP_VIDEO_AND_CLIPS = (
    "recommender-ucp-video-and-clips"
)


class LossFunctions(StrEnum):
    PAIR_LOGIT = "PairLogit"


class CatboostTaskTypes(StrEnum):
    CPU = "CPU"


class YtProxyClusterNames(StrEnum):
    """
    Pattern:
      - if auto():
          YT_PROXY_{cluster-name} -> {cluster-name}.yt.vk.team
          Example:
              YT_PROXY_JUPITER = auto() # jupiter.yt.vk.team
      - else:
          required explicit value
          Example:
              YT_PROXY_MIRANDA = "miranda.yt.vk.team"
    """

    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        _cluster_name = name.split("_")[-1].lower()
        _POSTFIX = ".yt.vk.team"
        return f"{_cluster_name}{_POSTFIX}"

    YT_PROXY_JUPITER = auto()
    YT_PROXY_SATURN = auto()
