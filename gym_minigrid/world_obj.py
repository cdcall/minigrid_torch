from gym_minigrid.constants import *
from abc import ABC, abstractmethod


class WorldObj(ABC):

    """
    Base class for grid world objects
    """

    def __init__(self, obj_type, color):
        assert obj_type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = obj_type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    @abstractmethod
    def can_contain(self):
        """Can this contain another object?"""
        pass  # return False

    @abstractmethod
    def can_overlap(self):
        """Can the agent overlap with this?"""
        pass  # return False

    @abstractmethod
    def can_pickup(self):
        """Can the agent pick this up?"""
        pass  # return False

    @abstractmethod
    def see_behind(self):
        """Can the agent see behind this object?"""
        pass  # return True

    @abstractmethod
    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        pass  # return False

    @abstractmethod
    def render(self, r):
        """Draw this object with the given renderer"""
        pass

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == 'wall':
            v = Wall(color)
        elif obj_type == 'floor':
            v = Floor(color)
        elif obj_type == 'ball':
            v = Ball(color)
        elif obj_type == 'key':
            v = Key(color)
        elif obj_type == 'box':
            v = Box(color)
        elif obj_type == 'door':
            v = Door(color, is_open, is_locked)
        elif obj_type == 'goal':
            v = Goal()
        elif obj_type == 'lava':
            v = Lava()
        elif obj_type == 'grass':
            v = Grass()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v


class Goal(WorldObj):

    def __init__(self):
        super().__init__('goal', 'green')

    def can_contain(self):
        return False

    def can_overlap(self):
        return True

    def can_pickup(self):
        return False

    def see_behind(self):
        return True

    def toggle(self, env, pos):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color='blue'):
        super().__init__('floor', color)

    def can_contain(self):
        return False  # True?

    def can_overlap(self):
        return True

    def can_pickup(self):
        return False

    def see_behind(self):
        return True

    def toggle(self, env, pos):
        return False

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):
    def __init__(self):
        super().__init__('lava', 'red')

    def can_contain(self):
        return False

    def can_overlap(self):
        return True  # TODO False

    def can_pickup(self):
        return False

    def see_behind(self):
        return True

    def toggle(self, env, pos):
        return False

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Grass(WorldObj):

    def __init__(self):
        super().__init__('grass', 'green')

    def can_contain(self):
        return False

    def can_overlap(self):
        return True  # TODO I think so

    def can_pickup(self):
        return False

    def see_behind(self):
        return True

    def toggle(self, env, pos):
        return False

    def render(self, img):
        c = (0, 255, 0)

        # Background color
        # fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little green waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), c)
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), c)
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), c)
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), c)


class Wall(WorldObj):
    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def can_contain(self):
        return False

    def can_overlap(self):
        return False

    def can_pickup(self):
        return False

    def see_behind(self):
        return False

    def toggle(self, env, pos):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Door(WorldObj):
    def __init__(self, color, is_open=False, is_locked=False):
        super().__init__('door', color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_contain(self):
        return False

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def can_pickup(self):
        return False

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        state = 0
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1

        return OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Key(WorldObj):
    def __init__(self, color='blue'):
        super(Key, self).__init__('key', color)

    def can_contain(self):
        return False

    def can_overlap(self):
        return False

    def can_pickup(self):
        return True

    def see_behind(self):
        return True

    def toggle(self, env, pos):
        return False

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):

    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def can_contain(self):
        return False

    def can_overlap(self):
        return False

    def can_pickup(self):
        return True

    def see_behind(self):
        return True

    def toggle(self, env, pos):
        return False

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Box(WorldObj):
    def __init__(self, color, contains=None):
        super(Box, self).__init__('box', color)
        self.contains = contains

    def can_contain(self):
        return False

    def can_overlap(self):
        return False

    def can_pickup(self):
        return True

    def see_behind(self):
        return True

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)
