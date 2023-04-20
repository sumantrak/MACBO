from .value_acquisition_strategy import (
    BOValueAcquisitionStrategy,
    MarginalDistValueAcquisitionStrategy,
    FixedValueAcquisitionStrategy,
    GridValueAcquisitionStrategy,
    LinspacePlot,
    UniformMaxValueAcquisitionStrategy
)
from collections import defaultdict
from envs.samplers import Constant

value_acquistion_strategies = {"gp-ucb": BOValueAcquisitionStrategy, "sample-dist": MarginalDistValueAcquisitionStrategy, "fixed": FixedValueAcquisitionStrategy, "grid": GridValueAcquisitionStrategy, "linspace-plot": LinspacePlot, "unimax":UniformMaxValueAcquisitionStrategy}


class AcquisitionStrategy(object):
    def __init__(self, model, env, args):
        self.model = model
        self.args = args
        self.num_samples = args.num_samples
        self.env = env
        self.value_strategy = args.value_strategy
        self.reward_node = args.reward_node

    def acquire(self, nodes, iteration):
        strategy = self.get_value_strategy(nodes)
        strategy(self._score_for_value, n_iters = self.args.num_intervention_values)

        selected_interventions = defaultdict(list)

        selected_interventions[strategy.max_j].extend([Constant(strategy.max_x)])

        return selected_interventions

    def get_value_strategy(self, nodes):
        strategy = value_acquistion_strategies[self.value_strategy](nodes = nodes, args = self.args)
        return strategy

