from enum import StrEnum, Enum, auto


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


class Metrics(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.upper()

    QUERY_AUC = auto()
    AUC_WEIGHTED = auto()
    QUERY_AUC_WEIGHTED = auto()
    AUC_WEIGHTED_NORMALIZED = auto()
    GAUC = auto()
