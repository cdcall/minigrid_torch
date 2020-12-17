from enum import IntEnum
from gym_minigrid.grid import Grid
from gym_minigrid.minigrid_env import MiniGridEnv
from gym.vector.utils import spaces
from gym_minigrid.world_obj import *
from abc import abstractmethod


class AirsimEnv(MiniGridEnv):

    """
    2D grid world game environment that simulates actions available in AirSim and Alpaca
    subclass of gym_minigrid MiniGridEnv which is an ABC
    """

    # Override superclass actions
    class NotActions(IntEnum):
        left = 0
        right = 1
        forward = 2
        # these 3 are ignored
        pickup = 3
        drop = 4
        toggle = 5
        # Done completing task
        done = 6
        # jump tiles
        jump2 = 7    # forward 2 tiles
        # jump3 = 8    # forward 3 tiles

    class Actions(IntEnum):
        left = 0
        right = 1
        forward = 2
        # Done completing task
        done = 3
        # jump tiles
        jump2 = 4    # forward 2 tiles
        # jump3 = 5    # forward 3 tiles

    def __init__(self, grid_size=None, width=None, height=None, max_steps=100, see_through_walls=False,
                 seed=1337, agent_view_size=7):

        super().__init__(grid_size, width, height, max_steps, see_through_walls, seed, agent_view_size)

        # Override superclass actions
        self.actions = self.Actions
        self.action_space = spaces.Discrete(len(self.actions))

        # Range of possible rewards  TODO
        self.reward_range = (-1, 1)

        # Range of possible penalties  TODO
        self.penalty_range = (0, 1)

    # -----------------------------------------------------
    # gym.Env basic agent methods
    # (step, reset, render, close, and seed)

    def step(self, action, logger=None):

        reward = 0
        penalty = 0
        done = False
        extra_steps = 0

        # TODO this is how it is done in gym-minigrid, but I'd be happier if this call was last
        # update dynamic object positions (if any)
        self._place_dynamic_obstacles()

        # Rotate left
        if action == self.actions.left:
            super()._rotate_left(logger)

        # Rotate right
        elif action == self.actions.right:
            super()._rotate_right(logger)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        # Move or jump forward (or an ignored action)
        elif action != self.actions.done:

            if action == self.actions.forward:
                if logger:
                    logger.info("    fwd")
                n_cells = 1

            elif action == self.actions.jump2:
                if logger:
                    logger.info("    jump2")
                n_cells = 2
                extra_steps = 1

            # elif action == self.actions.jump3:
            #     if logger:
            #         logger.info("    jump3")
            #     n_cells = 3
            #     extra_steps = 2

            else:
                n_cells = 0

            if n_cells:
                done, reward, penalty = self._move_forward(n_cells, logger)

        # error case
        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True
            if logger:
                logger.info("    max steps reached")

        obs = self.gen_obs()

        info_dict = {"penalty": penalty, "extra_steps": extra_steps}
        return obs, reward, done, info_dict

    # similar to superclass except highlight is False by default
    def render(self, mode='rgb_array', close=False, highlight=False, tile_size=TILE_PIXELS):
        if close:
            return
        highlight_mask = None
        if highlight:
            highlight_mask = self._create_highlight_mask()
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask,
            breadcrumbs=self.breadcrumbs
        )
        return img

    # superclass reset, close, and seed are fine as is

    # implement superclass abstract _gen_grid() method
    def _gen_grid(self, width, height):

        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self._place_goal()
        self._place_agent()
        self._place_static_obstacles()
        self._place_dynamic_obstacles()
        self._set_mission()

    def _move_forward(self, n_cells, logger):

        done = False
        reward = 0
        penalty = 0

        for i in range(n_cells):

            front_cell = self.grid.get(*self.front_pos)

            if front_cell and logger:
                logger.info(f"        front {front_cell.type}")

            # move forward one
            if not front_cell or front_cell.can_overlap():
                # self.step_count += 1
                super()._step_forward(logger)

            if front_cell and front_cell.type == 'goal':
                if logger:
                    logger.info(f"        -------------goal achieved in {self.step_count} steps\n\n")
                done = True
                reward = self._reward()
                break

            elif front_cell and front_cell.type == 'wall':
                # TODO penalty -- especially for jumping into a wall ?
                break

            elif front_cell and front_cell.type == 'grass':
                penalty = self._penalty()
                if logger:
                    logger.info(f"        -------------walked on grass at step {self.step_count}")

            elif front_cell and front_cell.type != 'goal':
                if logger:
                    logger.info(f"        -------------died at step {self.step_count} due to {front_cell.type}\n")
                reward = -1  # TODO this is how dynamic_env does it
                done = True
                break

        return done, reward, penalty

    # ------------------------------------------------------------
    # Airsim Env Abstract methods

    @abstractmethod
    def _place_goal(self):
        pass

    @abstractmethod
    def _place_agent(self):
        pass

    @abstractmethod
    def _place_static_obstacles(self):
        pass

    @abstractmethod
    def _place_dynamic_obstacles(self):
        pass

    @abstractmethod
    def _place_waypoint(self):
        pass

    @abstractmethod
    def _set_mission(self):
        pass

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        return 1 - 0.9 * (self.step_count / self.max_steps)

    @staticmethod
    def _penalty():
        # simple penalty as it's not dependent on which step and may occur multiple times
        return 0.25
