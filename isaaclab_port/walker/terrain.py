"""NumPy port of the official terrain generator, plus GPU lookup tables.

generate_course() draws the same random stream as the official environment, so a
(config, seed) pair gives the same course. Stairs are ported too, though the paper's
runs use only roughness, pits and stumps. compile_course() turns a course into the
flat arrays TerrainBank uploads, with no dependency on Isaac Lab or Box2D.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from local_runtime.legacy_seeding import np_random

from . import spec

STEP = spec.TERRAIN_STEP
COLLISION_THICKNESS = 0.3
HEIGHT_DX = 0.02
CELL = 0.5
# Hull-to-foot reach is at most 2.53 m, so this window always covers the legs.
COLLISION_REACH = 3.0
COLLISION_SLOTS = 28
LIDAR_SLOTS = 56
PARK_Z = -100.0


@dataclass(frozen=True)
class EnvConfig:
    """Obstacle encoding. Ranges are (low, high) in terrain steps, empty when absent."""
    ground_roughness: float = 0.0
    pit_gap: tuple = ()
    stump_width: tuple = ()
    stump_height: tuple = ()
    stump_float: tuple = ()
    stair_height: tuple = ()
    stair_width: tuple = ()
    stair_steps: tuple = ()

    @property
    def vector(self):
        """The official novelty encoding: roughness, pit range, stump height range."""
        pit = self.pit_gap or (0.0, 0.0)
        stump = self.stump_height or (0.0, 0.0)
        return np.array([self.ground_roughness, pit[0], pit[1], stump[0], stump[1]], dtype=float)

    @property
    def key(self):
        return json.dumps(asdict(self), sort_keys=True)

    @property
    def name(self):
        return "env_" + hashlib.sha256(self.key.encode()).hexdigest()[:10]


@dataclass
class Course:
    x: np.ndarray
    y: np.ndarray
    polygons: list
    initial_force: float


def generate_course(config, seed):
    rng, _ = np_random(int(seed))
    grass, stump, stairs, pit = 0, -1, -1, -1
    states = 1
    if config.stump_width and config.stump_height and config.stump_float:
        stump, states = states, states + 1
    if config.stair_height and config.stair_width and config.stair_steps:
        stairs, states = states, states + 1
    if config.pit_gap:
        pit, states = states, states + 1
    hardcore = states > 1

    state, velocity, y = grass, 0.0, spec.TERRAIN_HEIGHT
    counter, oneshot = spec.TERRAIN_STARTPAD, False
    xs, ys, polygons = [], [], []
    pit_diff = 0
    for i in range(spec.TERRAIN_LENGTH):
        x = i * STEP
        xs.append(x)
        if state == grass and not oneshot:
            velocity = 0.8 * velocity + 0.01 * np.sign(spec.TERRAIN_HEIGHT - y)
            if i > spec.TERRAIN_STARTPAD:
                velocity += rng.uniform(-1, 1) / spec.SCALE
            y += config.ground_roughness * velocity
        elif state == pit and oneshot:
            pit_gap = 1.0 + rng.uniform(*config.pit_gap)
            counter = np.ceil(pit_gap)
            pit_diff = counter - pit_gap
            poly = [(x, y), (x + STEP, y), (x + STEP, y - 4 * STEP), (x, y - 4 * STEP)]
            polygons.append(poly)
            polygons.append([(p[0] + STEP * pit_gap, p[1]) for p in poly])
            counter += 2
            original_y = y
        elif state == pit and not oneshot:
            y = original_y
            if counter > 1:
                y -= 4 * STEP
            if counter == 1:
                xs[-1] = xs[-1] - pit_diff * STEP
                pit_diff = 0
        elif state == stump and oneshot:
            stump_width = rng.randint(*config.stump_width)
            stump_height = rng.uniform(*config.stump_height)
            stump_float = rng.randint(*config.stump_float)
            counter = stump_width
            base = y + stump_float * STEP
            polygons.append([(x, base), (x + stump_width * STEP, base),
                             (x + stump_width * STEP, base + stump_height * STEP),
                             (x, base + stump_height * STEP)])
        elif state == stairs and oneshot:
            stair_height = rng.uniform(*config.stair_height)
            stair_slope = 1 if rng.rand() > 0.5 else -1
            stair_width = rng.randint(*config.stair_width)
            stair_steps = rng.randint(*config.stair_steps)
            original_y = y
            for s in range(stair_steps):
                top = y + (s * stair_height * stair_slope) * STEP
                bottom = y + (-stair_height + s * stair_height * stair_slope) * STEP
                left, right = x + (s * stair_width) * STEP, x + ((1 + s) * stair_width) * STEP
                polygons.append([(left, top), (right, top), (right, bottom), (left, bottom)])
            counter = stair_steps * stair_width + 1
        elif state == stairs and not oneshot:
            s = stair_steps * stair_width - counter
            n = s // stair_width
            y = original_y + (n * stair_height * stair_slope) * STEP - \
                (stair_height if stair_slope == -1 else 0) * STEP
        oneshot = False
        ys.append(y)
        counter -= 1
        if counter == 0:
            counter = rng.randint(spec.TERRAIN_GRASS // 2, spec.TERRAIN_GRASS)
            if state == grass and hardcore:
                state = rng.randint(1, states)
            else:
                state = grass
            oneshot = True
    # The official clouds are cosmetic but consume random numbers before the start force.
    for _ in range(spec.TERRAIN_LENGTH // 20):
        rng.uniform(0, spec.TERRAIN_LENGTH)
        for _ in range(5):
            rng.uniform(0, 5 * STEP)
            rng.uniform(0, 5 * STEP)
    force = float(rng.uniform(-spec.INITIAL_RANDOM, spec.INITIAL_RANDOM))
    return Course(np.array(xs), np.array(ys), polygons, force)


def _boxes(course):
    """Oriented boxes (cx, cz, half length, half thickness, cos, sin), solid below each edge."""
    boxes = []
    for x0, y0, x1, y1 in zip(course.x[:-1], course.y[:-1], course.x[1:], course.y[1:]):
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        c, s = dx / length, dy / length
        cx = (x0 + x1) / 2 + s * COLLISION_THICKNESS
        cz = (y0 + y1) / 2 - c * COLLISION_THICKNESS
        boxes.append((cx, cz, length / 2, COLLISION_THICKNESS, c, s))
    for poly in course.polygons:
        px, py = [p[0] for p in poly], [p[1] for p in poly]
        hx, hz = (max(px) - min(px)) / 2, (max(py) - min(py)) / 2
        if hx > 1e-6 and hz > 1e-6:
            boxes.append(((max(px) + min(px)) / 2, (max(py) + min(py)) / 2, hx, hz, 1.0, 0.0))
    boxes = np.array(boxes, dtype=np.float32)
    extent = boxes[:, 2] * np.abs(boxes[:, 4]) + boxes[:, 3] * np.abs(boxes[:, 5])
    order = np.argsort(boxes[:, 0] + extent, kind="stable")
    return boxes[order], (boxes[:, 0] - extent)[order], (boxes[:, 0] + extent)[order]


def _segments(course):
    """All surfaces the lidar can hit, as (x0, y0, x1, y1), sorted by right end."""
    segments = [(x0, y0, x1, y1) for x0, y0, x1, y1
                in zip(course.x[:-1], course.y[:-1], course.x[1:], course.y[1:])]
    for poly in course.polygons:
        segments += [(*poly[i], *poly[(i + 1) % 4]) for i in range(4)]
    segments = np.array(segments, dtype=np.float32)
    left, right = segments[:, [0, 2]].min(axis=1), segments[:, [0, 2]].max(axis=1)
    order = np.argsort(right, kind="stable")
    return segments[order], left[order], right[order]


def _height_table(course, cells):
    grid = np.arange(cells) * HEIGHT_DX
    heights = np.interp(grid, course.x, course.y)
    for poly in course.polygons:
        px = [p[0] for p in poly]
        top = max(p[1] for p in poly)
        inside = (grid >= min(px)) & (grid <= max(px))
        heights[inside] = np.maximum(heights[inside], top)
    return heights.astype(np.float32)


def table_sizes():
    course_x = spec.TERRAIN_LENGTH * STEP + 1.0
    return dict(cells=int(course_x / CELL) + 2, heights=int(course_x / HEIGHT_DX) + 2)


def compile_course(course, *, max_boxes=320, max_segments=480):
    """Flat tables for one course. Raises if a window would overflow its slots."""
    sizes = table_sizes()
    boxes, box_left, box_right = _boxes(course)
    segments, seg_left, seg_right = _segments(course)
    if len(boxes) > max_boxes or len(segments) > max_segments:
        raise ValueError(f"Course too complex: {len(boxes)} boxes, {len(segments)} segments")
    cell_left = np.arange(sizes["cells"]) * CELL
    box_start = np.searchsorted(box_right, cell_left - COLLISION_REACH)
    seg_start = np.searchsorted(seg_right, cell_left)
    for i, left in enumerate(cell_left):
        count = np.sum((box_right >= left - COLLISION_REACH) & (box_left <= left + CELL + COLLISION_REACH))
        # Sorted by right end, so a box in the window can sit past the first slots.
        last = np.nonzero(box_left <= left + CELL + COLLISION_REACH)[0]
        span = (last[-1] - box_start[i] + 1) if len(last) else 0
        if count > COLLISION_SLOTS or span > COLLISION_SLOTS:
            raise ValueError(f"Collision window overflow at x={left}: {count} boxes, span {span}")
        last = np.nonzero(seg_left <= left + CELL + spec.LIDAR_RANGE)[0]
        span = (last[-1] - seg_start[i] + 1) if len(last) else 0
        if span > LIDAR_SLOTS:
            raise ValueError(f"Lidar window overflow at x={left}: span {span}")
    padded_boxes = np.zeros((max_boxes, 6), dtype=np.float32)
    padded_boxes[:, 1], padded_boxes[:, 2:4], padded_boxes[:, 4] = PARK_Z, 0.01, 1.0
    padded_boxes[:len(boxes)] = boxes
    padded_segments = np.full((max_segments, 4), PARK_Z, dtype=np.float32)
    padded_segments[:len(segments)] = segments
    return dict(boxes=padded_boxes, box_start=box_start.astype(np.int32),
                segments=padded_segments, seg_start=seg_start.astype(np.int32),
                heights=_height_table(course, sizes["heights"]),
                initial_force=np.float32(course.initial_force))
