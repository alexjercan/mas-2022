import argparse

from base import Agent, Action, Perception
from representation import GridRelativeOrientation, GridOrientation
from hunting import HuntingEnvironment, WildLifeAgentData, WildLifeAgent
from enum import Enum
from communication import SocialAction, AgentMessage

import time, random


class ProbabilityMap(object):
    def __init__(self, existing_map=None):
        self.__internal_dict = {}

        if existing_map:
            for k, v in existing_map.list_actions():
                self.__internal_dict[k] = v

    def empty(self):
        if self.__internal_dict:
            return False

        return True

    def put(self, action, value):
        self.__internal_dict[action] = value

    def remove(self, action):
        """
        Updates a discrete action probability map by uniformly redistributing the probability of an action to remove over
        the remaining possible actions in the map.
        :param action: The action to remove from the map
        :return:
        """
        if action in self.__internal_dict:
            val = self.__internal_dict[action]
            del self.__internal_dict[action]

            remaining_actions = list(self.__internal_dict.keys())
            nr_remaining_actions = len(remaining_actions)

            if nr_remaining_actions != 0:
                prob_sum = 0
                for i in range(nr_remaining_actions - 1):
                    new_action_prob = (
                        self.__internal_dict[remaining_actions[i]] + val
                    ) / float(nr_remaining_actions)
                    prob_sum += new_action_prob

                    self.__internal_dict[remaining_actions[i]] = new_action_prob

                self.__internal_dict[remaining_actions[nr_remaining_actions - 1]] = (
                    1 - prob_sum
                )

    def choice(self):
        """
        Return a random action from a discrete distribution over a set of possible actions.
        :return: an action chosen from the set of choices
        """
        r = random.random()
        count_prob = 0

        for a in self.__internal_dict.keys():
            count_prob += self.__internal_dict[a]
            if count_prob >= r:
                return a

        raise RuntimeError("Should never get to this point when selecting an action")

    def list_actions(self):
        return self.__internal_dict.items()


class MyAction(Action, Enum):
    """
    Physical actions for wildlife agents.
    """

    # The agent must move north (up)
    NORTH = 0

    # The agent must move east (right).
    EAST = 1

    # The agent must move south (down).
    SOUTH = 2

    # The agent must move west (left).
    WEST = 3


class MyAgentPerception(Perception):
    """
    The perceptions of a wildlife agent.
    """

    def __init__(
        self, agent_position, obstacles, nearby_predators, nearby_prey, messages=None
    ):
        """
        Default constructor
        :param agent_position: agents's position.
        :param obstacles: visible obstacles
        :param nearby_predators: visible predators - given as tuple (agent_id, grid position)
        :param nearby_prey: visible prey - given as tuple (agent_id, grid_position)
        :param messages: incoming messages, may be None
        """
        self.agent_position = agent_position
        self.obstacles = obstacles
        self.nearby_predators = nearby_predators
        self.nearby_prey = nearby_prey

        if messages:
            self.messages = messages
        else:
            self.messages = []


class MyPrey(WildLifeAgent):
    """
    Implementation of the prey agent.
    """

    UP_PROB = 0.25
    LEFT_PROB = 0.25
    RIGHT_PROB = 0.25
    DOWN_PROB = 0.25

    def __init__(self):
        super(MyPrey, self).__init__(WildLifeAgentData.PREY)

    def response(self, perceptions):
        """
        :param perceptions: The perceptions of the agent at each step
        :return: The `Action' that your agent takes after perceiving the environment at each step
        """
        agent_pos = perceptions.agent_position
        probability_map = ProbabilityMap()
        probability_map.put(MyAction.NORTH, MyPrey.UP_PROB)
        probability_map.put(MyAction.SOUTH, MyPrey.DOWN_PROB)
        probability_map.put(MyAction.WEST, MyPrey.LEFT_PROB)
        probability_map.put(MyAction.EAST, MyPrey.RIGHT_PROB)

        for obstacle_pos in perceptions.obstacles:
            if agent_pos.get_distance_to(obstacle_pos) > 1:
                continue

            relative_orientation = agent_pos.get_simple_relative_orientation(
                obstacle_pos
            )
            if relative_orientation == GridRelativeOrientation.FRONT:
                probability_map.remove(MyAction.NORTH)

            elif relative_orientation == GridRelativeOrientation.BACK:
                probability_map.remove(MyAction.SOUTH)

            elif relative_orientation == GridRelativeOrientation.RIGHT:
                probability_map.remove(MyAction.EAST)

            elif relative_orientation == GridRelativeOrientation.LEFT:
                probability_map.remove(MyAction.WEST)

        ## save available moves
        available_moves = ProbabilityMap(existing_map=probability_map)

        ## examine actions which are unavailable because of predators
        for (_, predator_pos) in perceptions.nearby_predators:
            relative_pos = agent_pos.get_simple_relative_orientation(predator_pos)

            if relative_pos == GridRelativeOrientation.FRONT:
                probability_map.remove(MyAction.NORTH)

            elif relative_pos == GridRelativeOrientation.FRONT_LEFT:
                probability_map.remove(MyAction.NORTH)
                probability_map.remove(MyAction.WEST)

            elif relative_pos == GridRelativeOrientation.FRONT_RIGHT:
                probability_map.remove(MyAction.NORTH)
                probability_map.remove(MyAction.EAST)

            elif relative_pos == GridRelativeOrientation.LEFT:
                probability_map.remove(MyAction.WEST)

            elif relative_pos == GridRelativeOrientation.RIGHT:
                probability_map.remove(MyAction.EAST)

            elif relative_pos == GridRelativeOrientation.BACK:
                probability_map.remove(MyAction.SOUTH)

            elif relative_pos == GridRelativeOrientation.BACK_LEFT:
                probability_map.remove(MyAction.SOUTH)
                probability_map.remove(MyAction.WEST)

            elif relative_pos == GridRelativeOrientation.BACK_RIGHT:
                probability_map.remove(MyAction.SOUTH)
                probability_map.remove(MyAction.EAST)

        if not probability_map.empty():
            return probability_map.choice()
        else:
            return available_moves.choice()


class MyRandomPredator(WildLifeAgent):

    UP_PROB = 0.25
    LEFT_PROB = 0.25
    RIGHT_PROB = 0.25
    DOWN_PROB = 0.25

    def __init__(self, map_width=None, map_height=None):
        super(MyRandomPredator, self).__init__(WildLifeAgentData.PREDATOR)
        self.map_width = map_width
        self.map_height = map_height

    def response(self, perceptions):
        agent_pos = perceptions.agent_position
        probability_map = ProbabilityMap()
        probability_map.put(MyAction.NORTH, MyRandomPredator.UP_PROB)
        probability_map.put(MyAction.SOUTH, MyRandomPredator.DOWN_PROB)
        probability_map.put(MyAction.WEST, MyRandomPredator.LEFT_PROB)
        probability_map.put(MyAction.EAST, MyRandomPredator.RIGHT_PROB)

        for obstacle_pos in perceptions.obstacles:
            if agent_pos.get_distance_to(obstacle_pos) > 1:
                continue

            relative_orientation = agent_pos.get_simple_relative_orientation(
                obstacle_pos
            )
            if relative_orientation == GridRelativeOrientation.FRONT:
                probability_map.remove(MyAction.NORTH)

            elif relative_orientation == GridRelativeOrientation.BACK:
                probability_map.remove(MyAction.SOUTH)

            elif relative_orientation == GridRelativeOrientation.RIGHT:
                probability_map.remove(MyAction.EAST)

            elif relative_orientation == GridRelativeOrientation.LEFT:
                probability_map.remove(MyAction.WEST)

        return probability_map.choice()


class MySmartRandomPredator(WildLifeAgent):

    UP_PROB = 0.25
    LEFT_PROB = 0.25
    RIGHT_PROB = 0.25
    DOWN_PROB = 0.25

    def __init__(self, map_width=None, map_height=None):
        super(MySmartRandomPredator, self).__init__(WildLifeAgentData.PREDATOR)
        self.map_width = map_width
        self.map_height = map_height

    def response(self, perceptions):
        agent_pos = perceptions.agent_position
        probability_map = ProbabilityMap()
        probability_map.put(MyAction.NORTH, MySmartRandomPredator.UP_PROB)
        probability_map.put(MyAction.SOUTH, MySmartRandomPredator.DOWN_PROB)
        probability_map.put(MyAction.WEST, MySmartRandomPredator.LEFT_PROB)
        probability_map.put(MyAction.EAST, MySmartRandomPredator.RIGHT_PROB)

        for obstacle_pos in perceptions.obstacles:
            if agent_pos.get_distance_to(obstacle_pos) > 1:
                continue

            relative_orientation = agent_pos.get_simple_relative_orientation(
                obstacle_pos
            )
            if relative_orientation == GridRelativeOrientation.FRONT:
                probability_map.remove(MyAction.NORTH)

            elif relative_orientation == GridRelativeOrientation.BACK:
                probability_map.remove(MyAction.SOUTH)

            elif relative_orientation == GridRelativeOrientation.RIGHT:
                probability_map.remove(MyAction.EAST)

            elif relative_orientation == GridRelativeOrientation.LEFT:
                probability_map.remove(MyAction.WEST)

        ## examine actions which are better to do because of existing prey
        better_moves = []
        for (_, prey_pos) in perceptions.nearby_prey:
            relative_pos = agent_pos.get_simple_relative_orientation(prey_pos)

            if relative_pos == GridRelativeOrientation.FRONT:
                better_moves.append(MyAction.NORTH)

            elif relative_pos == GridRelativeOrientation.FRONT_LEFT:
                better_moves.append(MyAction.NORTH)
                better_moves.append(MyAction.WEST)

            elif relative_pos == GridRelativeOrientation.FRONT_RIGHT:
                better_moves.append(MyAction.NORTH)
                better_moves.append(MyAction.EAST)

            elif relative_pos == GridRelativeOrientation.LEFT:
                better_moves.append(MyAction.WEST)

            elif relative_pos == GridRelativeOrientation.RIGHT:
                better_moves.append(MyAction.EAST)

            elif relative_pos == GridRelativeOrientation.BACK:
                better_moves.append(MyAction.SOUTH)

            elif relative_pos == GridRelativeOrientation.BACK_LEFT:
                better_moves.append(MyAction.SOUTH)
                better_moves.append(MyAction.WEST)

            elif relative_pos == GridRelativeOrientation.BACK_RIGHT:
                better_moves.append(MyAction.SOUTH)
                better_moves.append(MyAction.EAST)

        if not better_moves:
            return probability_map.choice()

        bad_actions = [
            key for key, _ in probability_map.list_actions() if key not in better_moves
        ]
        for action in bad_actions:
            probability_map.remove(action)

        return probability_map.choice()


class MySmartCommunicativeRandomPredator(WildLifeAgent):

    UP_PROB = 0.25
    LEFT_PROB = 0.25
    RIGHT_PROB = 0.25
    DOWN_PROB = 0.25

    def __init__(self, map_width=None, map_height=None):
        super(MySmartCommunicativeRandomPredator, self).__init__(
            WildLifeAgentData.PREDATOR
        )
        self.map_width = map_width
        self.map_height = map_height

    def response(self, perceptions):
        agent_pos = perceptions.agent_position
        probability_map = ProbabilityMap()
        probability_map.put(MyAction.NORTH, MySmartCommunicativeRandomPredator.UP_PROB)
        probability_map.put(
            MyAction.SOUTH, MySmartCommunicativeRandomPredator.DOWN_PROB
        )
        probability_map.put(MyAction.WEST, MySmartCommunicativeRandomPredator.LEFT_PROB)
        probability_map.put(
            MyAction.EAST, MySmartCommunicativeRandomPredator.RIGHT_PROB
        )

        for obstacle_pos in perceptions.obstacles:
            if agent_pos.get_distance_to(obstacle_pos) > 1:
                continue

            relative_orientation = agent_pos.get_simple_relative_orientation(
                obstacle_pos
            )
            if relative_orientation == GridRelativeOrientation.FRONT:
                probability_map.remove(MyAction.NORTH)

            elif relative_orientation == GridRelativeOrientation.BACK:
                probability_map.remove(MyAction.SOUTH)

            elif relative_orientation == GridRelativeOrientation.RIGHT:
                probability_map.remove(MyAction.EAST)

            elif relative_orientation == GridRelativeOrientation.LEFT:
                probability_map.remove(MyAction.WEST)

        ## examine actions which are better to do because of existing prey
        better_moves = []
        known_prey_poses = []
        nearby_prey_pos = [prey_pos for _, prey_pos in perceptions.nearby_prey]
        prey_poses = nearby_prey_pos + AgentMessage.get_messages_content(AgentMessage.filter_messages_for(perceptions.messages, self))
        for prey_pos in prey_poses:
            known_prey_poses.append(prey_pos)
            relative_pos = agent_pos.get_simple_relative_orientation(prey_pos)

            if relative_pos == GridRelativeOrientation.FRONT:
                better_moves.append(MyAction.NORTH)

            elif relative_pos == GridRelativeOrientation.FRONT_LEFT:
                better_moves.append(MyAction.NORTH)
                better_moves.append(MyAction.WEST)

            elif relative_pos == GridRelativeOrientation.FRONT_RIGHT:
                better_moves.append(MyAction.NORTH)
                better_moves.append(MyAction.EAST)

            elif relative_pos == GridRelativeOrientation.LEFT:
                better_moves.append(MyAction.WEST)

            elif relative_pos == GridRelativeOrientation.RIGHT:
                better_moves.append(MyAction.EAST)

            elif relative_pos == GridRelativeOrientation.BACK:
                better_moves.append(MyAction.SOUTH)

            elif relative_pos == GridRelativeOrientation.BACK_LEFT:
                better_moves.append(MyAction.SOUTH)
                better_moves.append(MyAction.WEST)

            elif relative_pos == GridRelativeOrientation.BACK_RIGHT:
                better_moves.append(MyAction.SOUTH)
                better_moves.append(MyAction.EAST)

        if not better_moves:
            return probability_map.choice()

        bad_actions = [
            key for key, _ in probability_map.list_actions() if key not in better_moves
        ]
        for action in bad_actions:
            probability_map.remove(action)

        action = SocialAction(probability_map.choice())

        for known_prey_pos in known_prey_poses:
            for (predator_id, _) in perceptions.nearby_predators:
                action.add_outgoing_message(self.id, predator_id, known_prey_pos)

        return action


class MyEnvironment(HuntingEnvironment):
    """
    Your implementation of the environment in which cleaner agents work.
    """

    PREY_RANGE = 2
    PREDATOR_RANGE = 3

    def __init__(self, w, h, num_predators, num_prey, predator_type):
        """
        Default constructor. This should call the initialize methods offered by the super class.
        """
        rand_seed = 42
        # rand_seed = time.time()

        print("Seed = %i" % rand_seed)

        super(MyEnvironment, self).__init__()

        predators = []
        prey = []

        for i in range(num_predators):
            predators.append(predator_type(map_width=w, map_height=h))

        for i in range(num_prey):
            prey.append(MyPrey())

        """ Message box for messages that need to be delivered by the environment to their respective recepients, on
        the next turn """
        self.message_box = []

        ## initialize the huniting environment
        self.initialize(
            w=w, h=h, predator_agents=predators, prey_agents=prey, rand_seed=rand_seed
        )

        # Used to count how many steps it takes to finish
        self.steps = 0

    def step(self):
        """
        This method should iterate through all agents, provide them provide them with perceptions, and apply the
        action they return.
        """
        """
        STAGE 1: generate perceptions for all agents, based on the state of the environment at the beginning of this
        turn
        """
        agent_perceptions = {}

        ## get perceptions for prey agents
        for prey_data in self._prey_agents:
            nearby_obstacles = self.get_nearby_obstacles(
                prey_data.grid_position, MyEnvironment.PREY_RANGE
            )
            nearby_predators = self.get_nearby_predators(
                prey_data.grid_position, MyEnvironment.PREY_RANGE
            )
            nearby_prey = self.get_nearby_prey(
                prey_data.grid_position, MyEnvironment.PREY_RANGE
            )

            predators = [
                (ag_data.linked_agent.id, ag_data.grid_position)
                for ag_data in nearby_predators
            ]
            prey = [
                (ag_data.linked_agent.id, ag_data.grid_position)
                for ag_data in nearby_prey
            ]

            agent_perceptions[prey_data] = MyAgentPerception(
                agent_position=prey_data.grid_position,
                obstacles=nearby_obstacles,
                nearby_predators=predators,
                nearby_prey=prey,
            )

        ## create perceptions for predator agents, including messages in the `message_box`
        for predator_data in self._predator_agents:
            nearby_obstacles = self.get_nearby_obstacles(
                predator_data.grid_position, MyEnvironment.PREDATOR_RANGE
            )
            nearby_predators = self.get_nearby_predators(
                predator_data.grid_position, MyEnvironment.PREDATOR_RANGE
            )
            nearby_prey = self.get_nearby_prey(
                predator_data.grid_position, MyEnvironment.PREDATOR_RANGE
            )

            predators = [
                (ag_data.linked_agent.id, ag_data.grid_position)
                for ag_data in nearby_predators
            ]
            prey = [
                (ag_data.linked_agent.id, ag_data.grid_position)
                for ag_data in nearby_prey
            ]

            agent_perceptions[predator_data] = MyAgentPerception(
                agent_position=predator_data.grid_position,
                obstacles=nearby_obstacles,
                nearby_predators=predators,
                nearby_prey=prey,
                messages=self.message_box,
            )

        """
        STAGE 2: call response for each agent to obtain desired actions
        """
        agent_actions = {}
        ## get actions for all agents
        for prey_data in self._prey_agents:
            agent_actions[prey_data] = prey_data.linked_agent.response(
                agent_perceptions[prey_data]
            )

        for predator_data in self._predator_agents:
            agent_actions[predator_data] = predator_data.linked_agent.response(
                agent_perceptions[predator_data]
            )

        """
        STAGE 3: apply the agents' actions in the environment
        """
        for prey_data in self._prey_agents:
            if not prey_data in agent_actions:
                print("Agent %s did not opt for any action!" % str(prey_data))

            else:
                prey_action = agent_actions[prey_data]
                new_position = None

                if prey_action == MyAction.NORTH:
                    new_position = prey_data.grid_position.get_neighbour_position(
                        GridOrientation.NORTH
                    )
                elif prey_action == MyAction.SOUTH:
                    new_position = prey_data.grid_position.get_neighbour_position(
                        GridOrientation.SOUTH
                    )
                elif prey_action == MyAction.EAST:
                    new_position = prey_data.grid_position.get_neighbour_position(
                        GridOrientation.EAST
                    )
                elif prey_action == MyAction.WEST:
                    new_position = prey_data.grid_position.get_neighbour_position(
                        GridOrientation.WEST
                    )

                if not new_position in self._xtiles:
                    prey_data.grid_position = new_position
                else:
                    print("Agent %s tried to go through a wall!" % str(prey_data))

        for predator_data in self._predator_agents:
            if not predator_data in agent_actions:
                print("Agent %s did not opt for any action!" % str(predator_data))

            else:
                predator_action = agent_actions[predator_data]
                new_position = None

                ## handle case for a SocialAction instance
                self.message_box = []
                if isinstance(predator_action, SocialAction):
                    self.message_box.extend(predator_action.outgoing_messages)
                    predator_action = predator_action.action

                if predator_action == MyAction.NORTH:
                    new_position = predator_data.grid_position.get_neighbour_position(
                        GridOrientation.NORTH
                    )
                elif predator_action == MyAction.SOUTH:
                    new_position = predator_data.grid_position.get_neighbour_position(
                        GridOrientation.SOUTH
                    )
                elif predator_action == MyAction.EAST:
                    new_position = predator_data.grid_position.get_neighbour_position(
                        GridOrientation.EAST
                    )
                elif predator_action == MyAction.WEST:
                    new_position = predator_data.grid_position.get_neighbour_position(
                        GridOrientation.WEST
                    )

                if not new_position in self._xtiles:
                    predator_data.grid_position = new_position
                else:
                    print("Agent %s tried to go through a wall!" % str(predator_data))

        """
        At the end of the turn remove the dead prey
        """
        self.remove_dead_prey()

        # Increase the number of steps by one
        self.steps += 1


class Tester(object):

    NUM_PREDATORS = 4
    NUM_PREY = 10

    WIDTH = 15
    HEIGHT = 10

    DELAY = 0.1

    def __init__(self, predator_type):
        self.env = MyEnvironment(
            Tester.WIDTH,
            Tester.HEIGHT,
            Tester.NUM_PREDATORS,
            Tester.NUM_PREY,
            predator_type,
        )

    def make_steps(self):
        while not self.env.goals_completed():
            self.env.step()

            print(self.env)

            time.sleep(Tester.DELAY)

        # Show the score of the env
        print(self.env.steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        type=str,
        dest="predator_type",
        default="random",
        help="type of the predator: random, smart_random, smart_comm_random",
    )
    args = parser.parse_args()

    if args.predator_type == "random":
        predator_type = MyRandomPredator
    elif args.predator_type == "smart_random":
        predator_type = MySmartRandomPredator
    elif args.predator_type == "smart_comm_random":
        predator_type = MySmartCommunicativeRandomPredator

    tester = Tester(predator_type)
    tester.make_steps()
