from sympy import bell
from torch import isin
from environment import *
import time

class MyAgent(BlocksWorldAgent):

    def __init__(self, name: str, desired_state: BlocksWorld):
        super(MyAgent, self).__init__(name=name)

        self.desired_state = desired_state
        self.perception = None
        self.actions = []


    def response(self, perception: BlocksWorldPerception):
        self.revise_beliefs(perception.current_world)

        if not self.actions:
            self.actions = self.plan()

        return self.actions.pop(0)

    def revise_beliefs(self, perceived_world_state: BlocksWorld):
        self.perception = perceived_world_state

        if not self.actions:
            return

        # If a block was randomly moved on top of the stack that we build
        # then we have to stop and handle that first
        if isinstance(self.actions[0], Stack):
            action = self.actions[0]
            block = action.get_first_arg()
            p_block = action.get_second_arg()
            stack = self.perception.get_stack(p_block)
            if not stack.is_clear(p_block):
                self.actions = [PutDown(block)]

    def plan(self) -> List[BlocksWorldAction]:
        stacks = self.perception.get_stacks()
        d_stacks = self.desired_state.get_stacks()

        # Make sure that all unlocked blocks are in stacks of one, basically unstack everything
        for stack in stacks:
            # Already a single stack block, if that block is desired bottom, then lock it
            if stack.is_single_block():
                block = stack.get_top_block()
                d_stack = self.desired_state.get_stack(block)
                if d_stack.get_bottom_block() == block:
                    return [Lock(block)]
                continue

            # Ignore locked stacks
            block = stack.get_top_block()
            if stack.is_locked(block):
                continue

            # If there is a stack of more than one check if it is desired and lock it otherwise unstack
            below = stack.get_below(block)
            b_stack = self.desired_state.get_stack(below)
            d_stack = self.desired_state.get_stack(block)
            if b_stack == d_stack and stack.is_locked(below) and d_stack.get_below(block) == below:
                return [Lock(block)]
            return [Unstack(block, below), PutDown(block)]

        # Go through each desired stack and one by one add a new block on top and also lock them
        # All the initial blocks should be locked from last step
        for d_stack in d_stacks:
            p_block = d_stack.get_bottom_block()
            for block in d_stack.get_blocks()[1:]:
                # If the block is in stash, then wait otherwise get the stack
                if all([block not in stack.get_blocks() for stack in stacks]):
                    return [NoAction()]
                stack = self.perception.get_stack(block)

                # Ignore already matching blocks, which will be locked
                if stack.is_locked(block):
                    p_block = block
                    continue

                return [PickUp(block), Stack(block, p_block)]


        return [AgentCompleted()]


    def status_string(self):
        return str(self) + " : " + " ".join([str(a) for a in self.actions])



class Tester(object):
    STEP_DELAY = 0.5
    TEST_SUITE = "tests/0e-large/"

    EXT = ".txt"
    SI  = "si"
    SF  = "sf"

    DYNAMICITY = .5

    AGENT_NAME = "*A"

    def __init__(self):
        self._environment = None
        self._agents = []

        self._initialize_environment(Tester.TEST_SUITE)
        self._initialize_agents(Tester.TEST_SUITE)



    def _initialize_environment(self, test_suite: str) -> None:
        filename = test_suite + Tester.SI + Tester.EXT

        with open(filename) as input_stream:
            self._environment = DynamicEnvironment(BlocksWorld(input_stream=input_stream))


    def _initialize_agents(self, test_suite: str) -> None:
        filename = test_suite + Tester.SF + Tester.EXT

        agent_states = {}

        with open(filename) as input_stream:
            desires = BlocksWorld(input_stream=input_stream)
            agent = MyAgent(Tester.AGENT_NAME, desires)

            agent_states[agent] = desires
            self._agents.append(agent)

            self._environment.add_agent(agent, desires, None)

            print("Agent %s desires:" % str(agent))
            print(str(desires))


    def make_steps(self):
        print("\n\n================================================= INITIAL STATE:")
        print(str(self._environment))
        print("\n\n=================================================")

        completed = False
        nr_steps = 0

        while not completed:
            completed = self._environment.step()

            time.sleep(Tester.STEP_DELAY)
            print(str(self._environment))

            for ag in self._agents:
                print(ag.status_string())

            nr_steps += 1

            print("\n\n================================================= STEP %i completed." % nr_steps)

        print("\n\n================================================= ALL STEPS COMPLETED")





if __name__ == "__main__":
    tester = Tester()
    tester.make_steps()
