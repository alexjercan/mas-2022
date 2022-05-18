from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction

import numpy as np

class OtherAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(OtherAgent, self).__init__(agent_id)

    def specify_share(self, perception: CommonsPerception) -> float:
        return 0.5

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        K = perception.resource_remaining
        resource_shares = perception.resource_shares

        if K <= 0.0:
            consumption_adjustment = {agent: 0 for agent in resource_shares}
            return AgentAction(self.id, resource_share=resource_shares[self.id]*0.5, consumption_adjustment=consumption_adjustment, no_action=False)
        elif K > 0.25:
            consumption_adjustment = {agent: -0.1 * resource_shares[agent] for agent in resource_shares}
            consumption_adjustment[self.id] = 0
            consumption_adjustment[self.id] = -sum(consumption_adjustment.values())
            return AgentAction(self.id, resource_share=resource_shares[self.id], consumption_adjustment=consumption_adjustment, no_action=False)

        return AgentAction(self.id, no_action=True)

    def inform_round_finished(self, negotiation_round: int, perception: CommonsPerception):
        ## information sent to the agent once the current round (including all adjustment rounds) is finished
        pass

