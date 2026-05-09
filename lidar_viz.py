#!/usr/bin/env python3
"""
DUST Rover LIDAR Visualizer
Coordinate frame: X = forward, Y = left, Z = up  (units: cm)
Edit `sensor_values` below to visualize live readings.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ─── Edit these 17 values with live sensor data ───────────────────────────────
# -1 = no hit;  positive = distance to nearest obstacle in cm  (max ~1000 cm)
sensor_values = np.array([
    -1,   # 0  front-left wheel hub   | yaw 30° left
    350,  # 1  front-left frame       | yaw 20° left + pitch 20° down
    500,  # 2  front-center           | forward
    400,  # 3  front-right frame      | yaw 20° right + pitch 20° down
    -1,   # 4  front-right wheel hub  | yaw 30° right
    800,  # 5  front-left frame       | pitch 25° down
    750,  # 6  front-right frame      | pitch 25° down
    -1,   # 7  center-left frame      | left + pitch 20° down
    -1,   # 8  center-right frame     | right + pitch 20° down
    -1,   # 9  rear-left wheel hub    | backward + yaw 40° left
    -1,   # 10 rear-left frame        | backward
    -1,   # 11 rear-right frame       | backward
    -1,   # 12 rear-right wheel hub   | backward + yaw 40° right
    600,  # 13 front-left frame       | yaw 20° left + pitch 10° down
    550,  # 14 front-right frame      | yaw 20° right + pitch 10° down
    -1,   # 15 front-left wheel hub   | yaw 15° left
    -1,   # 16 front-right wheel hub  | yaw 15° right
], dtype=float)
# ──────────────────────────────────────────────────────────────────────────────

POSITIONS = np.array([
    [ 250,  245,  50],  # 0
    [ 325,   75, 130],  # 1
    [ 325,    0, 130],  # 2
    [ 325,  -75, 130],  # 3
    [ 250, -245,  50],  # 4
    [ 325,   75, 130],  # 5
    [ 325,  -75, 130],  # 6
    [  40,  235, 100],  # 7
    [  40, -235, 100],  # 8
    [-215,  270,  70],  # 9
    [-320,   80,  10],  # 10
    [-320,  -50,  10],  # 11
    [-215, -215,  70],  # 12
    [ 325,   75, 130],  # 13
    [ 325,  -75, 130],  # 14
    [ 250,  245,  50],  # 15
    [ 250, -245,  50],  # 16
], dtype=float)


def _yp(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """Unit direction from forward after yaw (CCW/left = +) then pitch (down = +)."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    # Intrinsic ZY rotation applied to (1,0,0) → (cosY·cosP, sinY·cosP, -sinP)
    return np.array([np.cos(y) * np.cos(p), np.sin(y) * np.cos(p), -np.sin(p)])


_r = np.radians
DIRECTIONS = np.array([
    _yp( 30),                                            # 0  yaw 30° left
    _yp( 20, 20),                                        # 1  yaw 20° L + pitch 20° dn
    _yp(  0),                                            # 2  forward
    _yp(-20, 20),                                        # 3  yaw 20° R + pitch 20° dn
    _yp(-30),                                            # 4  yaw 30° right
    _yp(  0, 25),                                        # 5  pitch 25° down
    _yp(  0, 25),                                        # 6  pitch 25° down
    [0,  np.cos(_r(20)), -np.sin(_r(20))],               # 7  left + pitch 20° down
    [0, -np.cos(_r(20)), -np.sin(_r(20))],               # 8  right + pitch 20° down
    [-np.cos(_r(40)),  np.sin(_r(40)), 0],               # 9  back + yaw 40° toward left
    [-1, 0, 0],                                          # 10 backward
    [-1, 0, 0],                                          # 11 backward
    [-np.cos(_r(40)), -np.sin(_r(40)), 0],               # 12 back + yaw 40° toward right
    _yp( 20, 10),                                        # 13 yaw 20° L + pitch 10° dn
    _yp(-20, 10),                                        # 14 yaw 20° R + pitch 10° dn
    _yp( 15),                                            # 15 yaw 15° left
    _yp(-15),                                            # 16 yaw 15° right
])

MAX_CM = 1000.0  # 10 m
_BG, _FG = '#12121f', 'white'


def _ray_endpoints(values: np.ndarray) -> np.ndarray:
    lengths = np.where(values < 0, MAX_CM, np.clip(values, 0, MAX_CM))
    return POSITIONS + DIRECTIONS * lengths[:, None]


def _box_edges():
    """12 edges of the rover body wireframe."""
    xs, ys, zs = [-320, 325], [-180, 180], [0, 150]
    c = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    pairs = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),
             (4,5),(4,6),(5,7),(6,7)]
    return [(c[a], c[b]) for a, b in pairs]


def _style_2d(ax, title, xl, yl):
    ax.set_facecolor('#0a0a18')
    ax.tick_params(colors=_FG, labelsize=7)
    ax.set_xlabel(xl, color=_FG, fontsize=8)
    ax.set_ylabel(yl, color=_FG, fontsize=8)
    ax.set_title(title, color=_FG, fontsize=9, pad=5)
    ax.grid(True, alpha=0.12, color='white', linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor('#444455')


def visualize(values: np.ndarray) -> None:
    ends = _ray_endpoints(values)

    fig = plt.figure(figsize=(18, 10), facecolor=_BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.22,
                            left=0.05, right=0.97, top=0.93, bottom=0.09)

    ax3  = fig.add_subplot(gs[:, 0], projection='3d')
    axXY = fig.add_subplot(gs[0, 1])
    axXZ = fig.add_subplot(gs[1, 1])
    axYZ = fig.add_subplot(gs[2, 1])

    fig.suptitle('DUST Rover LIDAR — Visualization', color=_FG,
                 fontsize=14, fontweight='bold')

    # 3D styling
    ax3.set_facecolor('#0a0a18')
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#2a2a4a')
    ax3.tick_params(colors=_FG, labelsize=6)
    ax3.set_xlabel('X fwd (cm)', color=_FG, fontsize=7, labelpad=2)
    ax3.set_ylabel('Y left (cm)', color=_FG, fontsize=7, labelpad=2)
    ax3.set_zlabel('Z up (cm)', color=_FG, fontsize=7, labelpad=2)
    ax3.set_title('3D View', color=_FG, fontsize=9, pad=6)

    _style_2d(axXY, 'Top View (XY)',          'X forward (cm)', 'Y left (cm)')
    _style_2d(axXZ, 'Side View (XZ)',         'X forward (cm)', 'Z up (cm)')
    _style_2d(axYZ, 'Front / Rear View (YZ)', 'Y left (cm)',    'Z up (cm)')

    # Rover body outlines
    box_kw = dict(color='#7777aa', alpha=0.30, linewidth=0.8, linestyle='--')
    for a, b in _box_edges():
        ax3.plot(*zip(a, b), **{**box_kw, 'zorder': 1})

    fp = np.array([[-320,-180],[325,-180],[325,180],[-320,180],[-320,-180]])
    axXY.plot(fp[:,0], fp[:,1], **box_kw)
    for wx, wy in [(250,245),(250,-245),(-215,270),(-215,-215)]:
        axXY.add_patch(patches.Circle((wx, wy), 45, fill=False,
                                      color='#7777aa', alpha=0.3, lw=0.7))
    axXY.annotate('▶ FWD', xy=(340, 0), color='#ffcc44', fontsize=8,
                  va='center', ha='left', alpha=0.8)

    axXZ.plot([-320, 325, 325,-320,-320], [0, 0, 150, 150, 0], **box_kw)
    axYZ.plot([-245, 245, 245,-245,-245], [0, 0, 150, 150, 0], **box_kw)

    # Rays
    for i in range(17):
        p, e  = POSITIONS[i], ends[i]
        hit   = values[i] >= 0
        col   = '#00ee77' if hit else '#2266ff'
        alpha = 0.90    if hit else 0.38

        ax3.plot([p[0],e[0]], [p[1],e[1]], [p[2],e[2]],
                 color=col, alpha=alpha, lw=1.5, zorder=3)
        ax3.scatter(*p, color='#ff9900', s=18, zorder=5, depthshade=False)
        if hit:
            ax3.scatter(*e, color='#ff3333', s=12, zorder=5, depthshade=False)
        ax3.text(e[0], e[1], e[2], str(i),
                 color='#ffff99', fontsize=6, ha='center', va='bottom', zorder=6)

        axXY.plot([p[0],e[0]], [p[1],e[1]], color=col, alpha=alpha, lw=1.2)
        axXY.scatter(p[0], p[1], color='#ff9900', s=16, zorder=5)
        if hit:
            axXY.scatter(e[0], e[1], color='#ff3333', s=10, zorder=5)
        # Push label 60 cm along the ray's XY projection to reduce overlap
        d2 = DIRECTIONS[i, :2]
        nm = np.linalg.norm(d2)
        off = (d2 / nm * 60) if nm > 0.05 else np.array([0.0, 10.0])
        axXY.text(p[0] + off[0], p[1] + off[1], str(i),
                  color='#ffff99', fontsize=6, ha='center', va='center', zorder=6)

        axXZ.plot([p[0],e[0]], [p[2],e[2]], color=col, alpha=alpha, lw=1.2)
        axXZ.scatter(p[0], p[2], color='#ff9900', s=16, zorder=5)
        if hit:
            axXZ.scatter(e[0], e[2], color='#ff3333', s=10, zorder=5)
        axXZ.text(e[0], e[2], str(i),
                  color='#ffff99', fontsize=6, ha='center', va='bottom', zorder=6)

        axYZ.plot([p[1],e[1]], [p[2],e[2]], color=col, alpha=alpha, lw=1.2)
        axYZ.scatter(p[1], p[2], color='#ff9900', s=16, zorder=5)
        if hit:
            axYZ.scatter(e[1], e[2], color='#ff3333', s=10, zorder=5)
        axYZ.text(e[1], e[2], str(i),
                  color='#ffff99', fontsize=6, ha='center', va='bottom', zorder=6)

    for ax in (axXY, axXZ, axYZ):
        ax.set_aspect('equal', adjustable='datalim')

    legend_elems = [
        Line2D([0],[0], color='#00ee77', lw=2,           label='Hit — returns distance'),
        Line2D([0],[0], color='#2266ff', lw=2, alpha=0.5, label='No hit — full 10 m shown'),
        Line2D([0],[0], marker='o', color='w', mfc='#ff9900', ms=7, lw=0, label='Sensor origin'),
        Line2D([0],[0], marker='o', color='w', mfc='#ff3333', ms=6, lw=0, label='Hit point'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=4, fontsize=8,
               facecolor='#1a1a2e', edgecolor='#555566', labelcolor=_FG,
               bbox_to_anchor=(0.5, 0.0))

    plt.show()


if __name__ == '__main__':
    visualize(sensor_values)
