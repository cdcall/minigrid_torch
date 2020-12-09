from gym_minigrid.minigrid_env import *
from gym_minigrid.register import register


class GrassEnv(MiniGridEnv):
    """
    Environment with one or more patches of grass which incur a penalty.
    """

    def __init__(self, size, obstacle_type=Grass, num_patches=1, seed=None):
        self.obstacle_type = obstacle_type

        # TODO check number of patches < some multiple of size

        if seed:
            seed = None
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=seed
        )

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        # TODO multiple grass patches
        # Generate and store random grass position
        self.grass_pos = np.array((
            self._rand_int(2, width - 2),
            self._rand_int(1, height - 1),
        ))

        # Place the patch of grass
        self.grid.vert_wall(self.grass_pos[0], self.grass_pos[1], 1, self.obstacle_type)

        self.mission = (
            "avoid penalties for stepping on the grass and get to the green goal square"
        )


class GrassS5Env(GrassEnv):
    def __init__(self):
        super().__init__(size=5)


class GrassS6Env(GrassEnv):
    def __init__(self):
        super().__init__(size=6)


class GrassS7Env(GrassEnv):
    def __init__(self):
        super().__init__(size=7)


register(
    id='MiniGrid-GrassS5-v0',
    entry_point='gym_minigrid.envs:GrassS5Env'
)

register(
    id='MiniGrid-GrassS6-v0',
    entry_point='gym_minigrid.envs:GrassS6Env'
)

register(
    id='MiniGrid-GrassS7-v0',
    entry_point='gym_minigrid.envs:GrassS7Env'
)
