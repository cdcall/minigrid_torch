from gym_minigrid.register import register
from operator import add
from navigation_env import *


class NavigationDynamicObstaclesEnv(NavigationEnv):
    """
    Single-room square grid environment with moving obstacles
    """
    def __init__(
            self,
            size=8,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            n_obstacles=4
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        # Reduce obstacles if there are too many
        if n_obstacles <= size/2 + 1:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size/2)

        self.obstacles = None

        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

    def _place_goal(self):
        # Place a goal square in the bottom-right corner
        self.grid.set(self.width - 2, self.height - 2, Goal())

    def _place_agent(self):
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def _place_static_obstacles(self):
        pass

    def _place_dynamic_obstacles(self):

        if not self.obstacles:
            # create the obstacles, and set initial position
            self.obstacles = []
            for i_obst in range(self.n_obstacles):
                self.obstacles.append(Ball())
                self.place_obj(self.obstacles[i_obst], max_tries=100)
                print(self.obstacles)

        # update obstacle positions
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            top = tuple(map(add, old_pos, (-1, -1)))
            try:
                # TODO not random placement!
                self.place_obj(self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100)
                self.grid.set(*old_pos, None)
            except:
                pass

    def _place_waypoint(self):
        pass

    def _set_mission(self):
        self.mission = "get to the green goal square"


class NavigationDynamicObstaclesEnv6x6(NavigationDynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=6, n_obstacles=3)


class NavigationDynamicObstaclesRandomEnv6x6(NavigationDynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None, n_obstacles=3)


class NavigationDynamicObstaclesEnv16x16(NavigationDynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=16, n_obstacles=8)


register(
    id='Navigation-Dynamic-Obstacles-6x6-v0',
    entry_point='navigation_envs:NavigationDynamicObstaclesEnv6x6'
)

register(
    id='Navigation-Dynamic-Obstacles-Random-6x6-v0',
    entry_point='navigation_envs:NavigationDynamicObstaclesRandomEnv6x6'
)

register(
    id='Navigation-Dynamic-Obstacles-16x16-v0',
    entry_point='navigation_envs:NavigationDynamicObstaclesEnv16x16'
)
