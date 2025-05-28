from enum import Enum

class SamplingMode(Enum):
    ACTOR_ROLLOUT = "actor_rollout"
    EXPERT_ROLLOUT = "expert_rollout"
    RELABEL_USING_EXPERT = "relabel_using_expert"
