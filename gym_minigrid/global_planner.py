# AStar Sample code in this file was taken from https://www.redblobgames.com/pathfinding/a-star/
# Copyright 2014 Red Blob Games <redblobgames@gmail.com>
#
# Feel free to use this code in your own projects, including commercial projects
# License: Apache v2.0 <http://www.apache.org/licenses/LICENSE-2.0.html>


from .world_obj import *
import math
from typing import Dict, List, Iterator, Tuple, TypeVar, Optional
import heapq

T = TypeVar('T')
Location = TypeVar('Location')
GridLocation = Tuple[int, int]


class GlobalPlanner:

    def __init__(self, 
                 use_carrot_stick=False, 
                 waypoint_dist=4):
        # If set to true, the waypoint will be provided as a carrot stick, i.e. # it will be updated every frame and will be always a fixed distance 
        # away from the agent and point it towards the goal. If set to False, 
        # waypoints will be provided as a fixed list of locations that lead to
        # the final goal
        self.use_carrot_stick = use_carrot_stick

        # The distance between 
        self.waypoint_dist = waypoint_dist

        self.grid = None

        # Current position of the agent
        self.agent_pos = None

        # Current position of the global goal
        self.goal_pos = None

        self.waypoint_pos = None

        self.square_grid = None
        
        self.planned_path : List[Location] = []
        # The original set of waypoints generated given the planned path
        self.waypoints : List[Location] = []
        self.waypoints_idx = []

        # The adjusted set of waypoints after accommodating for occupied 
        # grid cells where original waypoints could not be placed
        self.adjusted_waypoints : List[Location] = []

        # Acceptable variation in the location of the waypoints on the grid.
        # This is to resolve cases when waypoints cannot be placed in the
        # exact target location due to the existence of other (traversable)
        # objects in the same cell. This is used as the ration of 
        # self.waypoint_dist
        self.waypoint_loc_thresh = 0.5

        self.path_found = False
  
    def update_agent(self, agent_pos):
        self.agent_pos = agent_pos  

    def update_goal(self):
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                if (self.grid.get(i, j) and 
                    self.grid.get(i, j).type == 'goal'):
                    self.goal_pos = (i, j)
                    return 

    def load_map(self, grid):
        self.grid = grid
        self.square_grid = SquareGrid(self.grid.width, self.grid.height)
        self.square_grid.load_grid(self.grid)
        
    # Scan the map to update the obstacle locations
    def update_map(self):
        self.square_grid.load_grid(self.grid)

    def replan(self):
        # Remove the previously planned path and apply the changes to the grid
        self.remove_waypoints_from_grid()
        self.planned_path = []
        self.waypoints = []
        self.waypoints_idx = []
        self.adjusted_waypoints = []

        start = (self.agent_pos[0], self.agent_pos[1])
        goal = (self.goal_pos[0], self.goal_pos[1])
        came_from, cost = a_star_search(graph=self.square_grid, 
                          start=start, 
                          goal=goal)

        if not goal in came_from:
            # Path not found
            self.path_found = False
            return False
        else:
            self.path_found = True

        self.planned_path = reconstruct_path(came_from,
                            start=start, goal=goal) 

        self.waypoints = self.generate_waypoints()
        self.place_waypoints_on_grid()

        # draw_grid(self.square_grid, path=self.planned_path,
        #           point_to=came_from, start=start, goal=goal)                    
        
        return True   

    def get_path(self):
        return self.planned_path

    def get_waypoints(self):
        return self.waypoints

    # Selects a single or a set of waypoints (given the operation mode) after a 
    # path is planned
    def generate_waypoints(self):
        if not self.path_found:
            return []

        if self.use_carrot_stick:
            waypoint_idx = min(len(self.planned_path) - 2, self.waypoint_dist)

            if len(self.planned_path) - 1 <= self.waypoint_dist:
                self.waypoints_idx = []
                return []
            else:
                self.waypoints_idx = [self.waypoint_dist]
                return [self.planned_path[self.waypoint_dist]]
        else:
            waypoints : List[Location] = []
            waypoint_idx = range(self.waypoint_dist, 
                                 len(self.planned_path)-1, 
                                 self.waypoint_dist)
            for idx in waypoint_idx:
                waypoints.append(self.planned_path[idx])

            self.waypoints_idx = [idx for idx in waypoint_idx]
            return waypoints

    def remove_waypoints_from_grid(self):
        for waypoint in self.adjusted_waypoints:
            obj = self.grid.get(waypoint[0], waypoint[1])
            if obj:
                if obj.type == "waypoint":
                    self.grid.set(waypoint[0], waypoint[1], None)

    def place_waypoints_on_grid(self):
        for waypoint, idx in zip(self.waypoints, self.waypoints_idx):
            assert self.square_grid.in_bounds(waypoint)
            if self.grid.get(waypoint[0], waypoint[1]):
                self.place_waypoint_nearby(idx,
                    math.floor(self.waypoint_loc_thresh * self.waypoint_dist))
            else:
                self.grid.set(waypoint[0], waypoint[1], WayPoint())
                self.adjusted_waypoints.append(waypoint)

    # Place a waypoint in the nearest free location on the map to the target 
    # waypoint location given a distance threshold
    def place_waypoint_nearby(self, waypoint_idx, threshold):
        assert waypoint_idx - threshold > 0
        
        # The region of interest to search for a free cell to place the waypoint
        roi_start = waypoint_idx - threshold
        roi_end = min(waypoint_idx + threshold, len(self.planned_path) - 2)
        
        if roi_end - roi_start < 2:
            return False

        for idx in range(waypoint_idx - 1, roi_start - 1, -1):
            loc = self.planned_path[idx]
            if not self.grid.get(loc[0], loc[1]):
                self.grid.set(loc[0], loc[1], WayPoint())
                self.adjusted_waypoints.append(loc)
                return True
        for idx in range(waypoint_idx + 1, roi_end + 1):
            loc = self.planned_path[idx]
            if not self.grid.get(loc[0], loc[1]):
                self.grid.set(loc[0], loc[1], WayPoint())
                self.adjusted_waypoints.append(loc)
                return True

        return False


class SquareGrid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.obstacles: List[GridLocation] = []
        self.weights: Dict[GridLocation, float] = {}
        self.grid = None

    def load_grid(self, grid):
        assert grid.width > 0
        assert grid.height > 0
        self.grid = grid
        self.obstacles = []
        for i in range(self.width):
            for j in range(self.height):
                obj = grid.get(i, j)
                if obj:
                    if not obj.can_traverse():
                        self.obstacles.append((i, j))

    
    def in_bounds(self, id: GridLocation) -> bool:
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id: GridLocation) -> bool:
        return id not in self.obstacles
    
    def neighbors(self, id: GridLocation) -> Iterator[GridLocation]:
        (x, y) = id
        neighbors = [(x+1, y), (x-1, y), (x, y-1), (x, y+1)] # E W N S
        results = filter(self.in_bounds, neighbors)
        results = filter(self.passable, results)
        return results

    def cost(self, from_node: GridLocation, to_node: GridLocation) -> float:
        # TODO: Currently it is assumed that the traversal cost to
        # all neighbors is the same. We might want to consider the 
        # cost associated with different types of terrain or rotaion
        # actions
        return 1.0

class PriorityQueue:
    def __init__(self):
        self.elements: List[Tuple[float, T]] = []
    
    def empty(self) -> bool:
        return not self.elements
    
    def put(self, item: T, priority: float):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self) -> T:
        return heapq.heappop(self.elements)[1]



def a_star_search(graph: SquareGrid, start: Location, goal: Location):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from: Dict[Location, Optional[Location]] = {}
    cost_so_far: Dict[Location, float] = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current: Location = frontier.get()
        
        if current == goal:
            break

        for next in graph.neighbors(current):  
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far

def heuristic(a: GridLocation, b: GridLocation) -> float:
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from: Dict[Location, Location],
                     start: Location, goal: Location) -> List[Location]:
    current: Location = goal
    path: List[Location] = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path

# Utility functions for dealing with square grids

def draw_tile(graph, id, style):
    r = " . "
    if 'number' in style and id in style['number']: r = " %-2d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        if x2 == x1 + 1: r = " > "
        if x2 == x1 - 1: r = " < "
        if y2 == y1 + 1: r = " v "
        if y2 == y1 - 1: r = " ^ "
    if 'path' in style and id in style['path']:   r = " @ "
    if 'start' in style and id == style['start']: r = " A "
    if 'goal' in style and id == style['goal']:   r = " Z "
    if id in graph.obstacles: r = "###"
    return r

def draw_grid(graph, **style):
    print("___" * graph.width)
    for y in range(graph.height):
        for x in range(graph.width):
            print("%s" % draw_tile(graph, (x, y), style), end="")
        print()
    print("~~~" * graph.width)