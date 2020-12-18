from gym_minigrid.register import register
from gym_minigrid.world_obj import Grass, Goal
import numpy as np
from navigation_env import NavigationEnv


class GrassEnv(NavigationEnv):
    """
    Environment with one or more patches of grass which incur a penalty.
    """

    def __init__(self, size, obstacle_type=Grass, n_patches=1):

        self.obstacle_type = obstacle_type

        # Reduce grass patches if there are too many
        if n_patches <= size/2 + 1:
            self.n_patches = int(n_patches)
        else:
            self.n_patches = int(size/2)

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            use_global_planner=True,
            use_carrot_stick_waypoint=False,
            waypoint_distance=3
        )

    def _place_goal(self):
        # Place a goal square in the bottom-right corner
        self.grid.set(self.width - 2, self.height - 2, Goal())

    def _place_agent(self):
        # Place the agent in the top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0

    def _place_static_obstacles(self):
        # Generate and store random grass positions
        self.grass_pos = np.array((
            self._rand_int(1, self.width - 1),
            self._rand_int(1, self.height - 1),
        ))
        # place grass patches
        self.patches = []
        for i_patch in range(self.n_patches):
            self.patches.append(Grass())
            self.place_obj(self.patches[i_patch], max_tries=100)

    def _place_dynamic_obstacles(self):
        pass

    def _place_waypoint(self):
        pass

    def _set_mission(self):
        self.mission = (
            "avoid penalties for stepping on the grass and get to the green goal square"
        )


class NavigationGrassS5Env(GrassEnv):
    def __init__(self):
        super().__init__(size=5,  n_patches=1)


class NavigationGrassS6Env(GrassEnv):
    def __init__(self):
        super().__init__(size=6, n_patches=2)


class NavigationGrassS8Env(GrassEnv):
    def __init__(self):
        super().__init__(size=8, n_patches=4)


register(
    id='Navigation-GrassS5-v0',
    entry_point='navigation_envs:NavigationGrassS5Env'
)

register(
    id='Navigation-GrassS6-v0',
    entry_point='navigation_envs:NavigationGrassS6Env'
)

register(
    id='Navigation-GrassS8-v0',
    entry_point='navigation_envs:NavigationGrassS8Env'
)
