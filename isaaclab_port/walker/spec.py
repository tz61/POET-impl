"""Physical constants of the official Box2D walker, in its own (scaled) units.

Masses and inertias were measured from the official environment with Box2D 2.3.8;
tests/test_walker_port.py rechecks them when Box2D is installed. Hull inertia is
about its centre of mass, converted from Box2D's value about the body origin.
"""
import math

SCALE = 30.0
FPS = 50
GRAVITY = 10.0
MOTORS_TORQUE = 80.0
SPEED_HIP = 4.0
SPEED_KNEE = 6.0
LIDAR_RANGE = 160.0 / SCALE
LIDAR_ANGLES = tuple(1.5 * i / 10.0 for i in range(10))
VIEWPORT_W, VIEWPORT_H = 600.0, 400.0

TERRAIN_STEP = 14.0 / SCALE
TERRAIN_LENGTH = 200
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4.0
TERRAIN_GRASS = 10
TERRAIN_STARTPAD = 20
FINISH_X = (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP

HULL_POLY = tuple((x / SCALE, y / SCALE) for x, y in ((-30, 9), (6, 9), (34, 1), (34, -8), (-30, -8)))
LEG_DOWN = -8.0 / SCALE
LEG_W, LEG_H = 8.0 / SCALE, 34.0 / SCALE
LOWER_W = 0.8 * LEG_W

HULL_MASS = 5.42222261428833
HULL_COM = (-0.02003643661737442, -0.005646640434861183)
HULL_INERTIA = 2.0017447471618652 - HULL_MASS * (HULL_COM[0] ** 2 + HULL_COM[1] ** 2)
UPPER_MASS, UPPER_INERTIA = 0.30222222208976746, 0.03413992002606392
LOWER_MASS, LOWER_INERTIA = 0.24177777767181396, 0.026796143501996994

HIP_LIMITS = (-0.8, 1.1)
KNEE_LIMITS = (-1.6, -0.1)
# Box2D friction mixes as sqrt(a * b): legs 0.2, terrain 2.5.
LEG_FRICTION = math.sqrt(0.2 * 2.5)

# Box2D starts with violated joints that its solver snaps closed over two steps.
# This is that settled state, reached after reset plus one zero-action step:
# hull (x, y, angle, vx, vy, omega), then hips, knees and their speeds, per leg.
START_HULL = (4.647395610809326, 5.732183933258057, -0.005245721433311701,
              -0.03, -0.01728838123381138, 0.0004751558299176395)
START_HIP = (0.43643951416015625, 0.33912962675094604)
START_KNEE = (-0.8859171867370605, -0.8842307329177856)
START_HIP_SPEED = (0.285423219203949, 0.29809027910232544)
START_KNEE_SPEED = (-0.5273939371109009, -0.5533720254898071)
# Only the hull receives the initial random force, for one step.
INITIAL_RANDOM = 5.0
INITIAL_FORCE_TO_VX = (1.0 / FPS) / HULL_MASS
