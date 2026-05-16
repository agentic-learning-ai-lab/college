from dataclasses import dataclass


@dataclass
class AggregatorConfig:
    agg_method: str = "mean"
