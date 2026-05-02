from .base import EGGROLLStrategy
from .standard import StandardEGGROLL
from .greedy import GreedyEGGROLL
from .greedy_local import GreedyLocalEGGROLL
from .elitist import ElitistEGGROLL
from .sequential import SequentialEGGROLL
from .sequential_elitist import SequentialElitistEGGROLL
from .layer_grouped import LayerGroupedEGGROLL

STRATEGY_REGISTRY = {
    'standard': StandardEGGROLL,
    'greedy': GreedyEGGROLL,
    'greedy_local': GreedyLocalEGGROLL,
    'elitist': ElitistEGGROLL,
    'sequential': SequentialEGGROLL,
    'sequential_elitist': SequentialElitistEGGROLL,
    'layer_grouped': LayerGroupedEGGROLL,
}
