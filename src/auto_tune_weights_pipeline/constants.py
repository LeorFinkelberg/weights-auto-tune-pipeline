from enum import StrEnum, auto


RANDOM_SEED = 42424242
BIG_NEGATIVE_DEFAULT_VALUE = -3.4e38
DICTIONARY_HUB_URL = "https://dictionary-hub.kaizen.idzn.ru/fetch"
DICTIONARY_PROJECT_NAME_RECOMMENDER_UCP_VIDEO_AND_CLIPS = (
    "recommender-ucp-video-and-clips"
)


class Platforms(StrEnum):
    def _generate_next_value_(name, start, count, last_values) -> str:
        return name.lower()

    ANDROID = auto()
    VK_VIDEO_ANDROID = auto()


class NavScreens(StrEnum):
    def _generate_next_value_(name, start, count, last_values) -> str:
        return name.lower()

    VIDEO_FOR_YOU = auto()


class LossFunctions(StrEnum):
    PAIR_LOGIT = "PairLogit"
    PAIR_LOGIT_PAIRWISE = "PairLogitPairwise"


class CatboostTaskTypes(StrEnum):
    CPU = "CPU"


class SummaryLogFields(StrEnum):
    GAUC_SIMPLE = "GAUCSimple"
    GAUC_WEIGHTED = "GAUCWeighted"
    N_GROUPS = "n_groups"
    STD = "std"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    GROUP_DETAILS = "group_details"
    GAUC_VALID = "GAUC_valid"

    @classmethod
    def get_values(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)


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
