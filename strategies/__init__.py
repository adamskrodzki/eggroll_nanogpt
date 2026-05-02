from .base import EGGROLLStrategy
from .standard import StandardEGGROLL
from .greedy import GreedyEGGROLL
from .greedy_local import GreedyLocalEGGROLL
from .elitist import ElitistEGGROLL
from .sequential import SequentialEGGROLL

STRATEGY_REGISTRY = {
    'standard': StandardEGGROLL,
    'greedy': GreedyEGGROLL,
    'greedy_local': GreedyLocalEGGROLL,
    'elitist': ElitistEGGROLL,
    'sequential': SequentialEGGROLL,
}
