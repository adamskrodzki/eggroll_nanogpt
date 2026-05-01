from .base import EGGROLLStrategy
from .standard import StandardEGGROLL
from .greedy import GreedyEGGROLL
from .greedy_local import GreedyLocalEGGROLL
from .elitist import ElitistEGGROLL

STRATEGY_REGISTRY = {
    'standard': StandardEGGROLL,
    'greedy': GreedyEGGROLL,
    'greedy_local': GreedyLocalEGGROLL,
    'elitist': ElitistEGGROLL,
}
