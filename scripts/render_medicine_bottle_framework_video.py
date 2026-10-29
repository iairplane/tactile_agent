#!/usr/bin/env python3
"""Render the medicine-bottle framework explainer from recorded evidence.

The video deliberately distinguishes three authorities:

* cyan: free-space agentic planning;
* gold: object-relative contact motion;
* magenta: the task-session tactile supervisor.

The run-16 tactile overlay is reconstructed from ``action_log.json``.  It is
labelled as a relative raw-Fz proxy because the recording does not contain a
synchronised tactile RGB stream and the SDK wrench units were not physically
qualified for this installation.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1280
HEIGHT = 720
FPS = 30

BG = (5, 10, 18)
PANEL = (12, 21, 33)
PANEL_2 = (17, 29, 44)
WHITE = (241, 246, 252)
MUTED = (145, 164, 184)
GRID = (42, 58, 76)
CYAN = (38, 214, 229)
GOLD = (246, 184, 69)
PINK = (244, 87, 174)
GREEN = (65, 222, 151)
RED = (246, 92, 92)

FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_MEDIUM = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
FONT_BLACK = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc")
FONT_MONO = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")


@dataclass(frozen=True)
class Sources:
    demo: Path
    robot: Path
    localization: Path
    cap_mask: Path
    action_log: Path


class VideoReader:
    """Monotonic source-time reader with one seek per rendered scene."""

    def __init__(self, path: Path, start_s: float) -> None:
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.current_index = max(0, int(math.floor(start_s * self.fps)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_index)
        self.last: np.ndarray | None = None

    def frame_at(self, source_s: float) -> Image.Image:
        target = max(0, int(round(source_s * self.fps)))
        if target < self.current_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.current_index = target
            self.last = None
        while self.current_index <= target:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.last = frame
            self.current_index += 1
        if self.last is None:
            raise RuntimeError(f"failed to read {source_s:.3f}s")
        return Image.fromarray(cv2.cvtColor(self.last, cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        self.cap.release()


def font(size: int, *, black: bool = False, medium: bool = False, mono: bool = False):
    path = FONT_MONO if mono else FONT_BLACK if black else FONT_MEDIUM if medium else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def smooth(value: float) -> float:
    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


def fade_for(local_t: float, duration: float, edge: float = 0.35) -> float:
    return min(smooth(local_t / edge), smooth((duration - local_t) / edge))


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    size: int,
    color=WHITE,
    *,
    black: bool = False,
    medium: bool = False,
    mono: bool = False,
    anchor: str | None = None,
) -> None:
    draw.text(xy, text, font=font(size, black=black, medium=medium, mono=mono), fill=color, anchor=anchor)


def rounded_panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    fill=PANEL,
    outline=GRID,
    radius: int = 22,
    width: int = 1,
) -> None:
    ImageDraw.Draw(image).rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    *,
    size: int = 20,
    fill_alpha: int = 45,
) -> tuple[int, int, int, int]:
    f = font(size, medium=True)
    bbox = draw.textbbox((0, 0), text, font=f)
    w = bbox[2] - bbox[0] + 30
    h = bbox[3] - bbox[1] + 18
    x, y = xy
    fill = tuple(int((c * fill_alpha + PANEL_2[i] * (255 - fill_alpha)) / 255) for i, c in enumerate(color))
    draw.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=fill, outline=color, width=2)
    draw.text((x + w / 2, y + h / 2 - 1), text, font=f, fill=color, anchor="mm")
    return x, y, x + w, y + h


def contain(source: Image.Image, box: tuple[int, int, int, int], *, radius: int = 22) -> Image.Image:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    scale = min(bw / source.width, bh / source.height)
    size = (max(1, int(source.width * scale)), max(1, int(source.height * scale)))
    resized = source.resize(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (bw, bh), PANEL)
    px = (bw - size[0]) // 2
    py = (bh - size[1]) // 2
    panel.paste(resized, (px, py))
    mask = Image.new("L", (bw, bh), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, bw - 1, bh - 1), radius=radius, fill=255)
    out = Image.new("RGB", (WIDTH, HEIGHT), BG)
    out.paste(panel, (x0, y0), mask)
    return out


def cover(source: Image.Image, box: tuple[int, int, int, int], *, radius: int = 22) -> Image.Image:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    scale = max(bw / source.width, bh / source.height)
    size = (max(1, int(source.width * scale)), max(1, int(source.height * scale)))
    resized = source.resize(size, Image.Resampling.LANCZOS)
    left = (size[0] - bw) // 2
    top = (size[1] - bh) // 2
    crop = resized.crop((left, top, left + bw, top + bh))
    mask = Image.new("L", (bw, bh), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, bw - 1, bh - 1), radius=radius, fill=255)
    out = Image.new("RGB", (WIDTH, HEIGHT), BG)
    out.paste(crop, (x0, y0), mask)
    return out


def paste_layer(base: Image.Image, layer: Image.Image) -> None:
    base.paste(layer, (0, 0), layer if layer.mode == "RGBA" else None)


def top_chrome(image: Image.Image, chapter: str, color: tuple[int, int, int], progress: float) -> None:
    draw = ImageDraw.Draw(image)
    draw_text(draw, (42, 28), "TACTILE AGENT  /  ONE-SHOT MANIPULATION", 18, MUTED, medium=True)
    draw_text(draw, (1238, 29), chapter, 19, color, medium=True, anchor="ra")
    draw.rounded_rectangle((42, 58, 1238, 64), radius=3, fill=(24, 35, 49))
    draw.rounded_rectangle((42, 58, 42 + int(1196 * clamp(progress)), 64), radius=3, fill=color)


def subtitle(image: Image.Image, text: str, accent: tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 650, 1240, 704), radius=18, fill=(7, 14, 24), outline=(34, 49, 67), width=1)
    draw.rounded_rectangle((55, 665, 61, 689), radius=3, fill=accent)
    draw_text(draw, (82, 676), text, 24, WHITE, medium=True, anchor="lm")


def metric_card(image: Image.Image, box: tuple[int, int, int, int], value: str, label: str, color) -> None:
    rounded_panel(image, box, fill=PANEL_2, outline=tuple(int(c * 0.65) for c in color), radius=18, width=2)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, _ = box
    draw_text(draw, ((x0 + x1) / 2, y0 + 34), value, 31, color, black=True, mono=True, anchor="mm")
    draw_text(draw, ((x0 + x1) / 2, y0 + 70), label, 17, MUTED, anchor="mm")


def dim(image: Image.Image, opacity: int = 75) -> None:
    shade = Image.new("RGBA", image.size, (0, 0, 0, opacity))
    image.paste(shade, (0, 0), shade)


def stage_for_demo(source_t: float) -> tuple[str, tuple[int, int, int], int]:
    stages = [
        (0.60, 2.133, "APPROACH", CYAN, 0),
        (2.133, 2.80, "GRASP + PRESS", GOLD, 1),
        (2.80, 8.667, "TWIST + REGRASP", GOLD, 2),
        (8.667, 9.20, "RELEASE CHECK", PINK, 3),
        (9.20, 10.333, "LIFT + PLACE", GREEN, 4),
    ]
    for start, end, name, color, index in stages:
        if start <= source_t < end:
            return name, color, index
    return "ONE DEMONSTRATION", WHITE, 0


def render_hook(t: float, duration: float, reader: VideoReader) -> Image.Image:
    source_t = 219.35 + (227.4 - 219.35) * clamp(t / duration)
    frame = reader.frame_at(source_t)
    image = cover(frame, (0, 0, WIDTH, HEIGHT), radius=0)
    dim(image, 72)
    draw = ImageDraw.Draw(image)
    a = smooth(t / 0.65)
    y = int(208 + (1.0 - a) * 28)
    draw_text(draw, (640, 122), "ONE DEMO  →  CLOSED-LOOP SKILL", 20, CYAN, medium=True, mono=True, anchor="mm")
    draw_text(draw, (640, y), "一次示范，机器人自主旋开安全瓶盖", 48, WHITE, black=True, anchor="mm")
    draw_text(draw, (640, y + 68), "意图拆解  ×  物体相对位姿  ×  触觉监工", 27, (204, 216, 229), medium=True, anchor="mm")
    metric_card(image, (210, 420, 480, 518), "540°", "连续旋转", GOLD)
    metric_card(image, (505, 420, 775, 518), "5×", "自动回腕重抓", CYAN)
    metric_card(image, (800, 420, 1070, 518), "PASS", "release_verified", GREEN)
    pill(draw, (498, 564), "真实 FR3 + Hand-E + 双 GF220", GREEN, size=18)
    return image


def render_demo(t: float, duration: float, reader: VideoReader) -> Image.Image:
    source_t = 0.60 + (10.333 - 0.60) * clamp(t / duration)
    frame = reader.frame_at(source_t)
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    paste_layer(image, contain(frame, (42, 84, 880, 622), radius=22))
    top_chrome(image, "01  HUMAN DEMONSTRATION", CYAN, t / duration)
    draw = ImageDraw.Draw(image)
    name, color, active = stage_for_demo(source_t)
    pill(draw, (69, 104), name, color, size=18)
    rounded_panel(image, (910, 90, 1238, 622), fill=PANEL, outline=GRID)
    draw_text(draw, (938, 126), "示范被读成阶段", 25, WHITE, black=True)
    nodes = [
        ("靠近", "自由空间意图", CYAN),
        ("抓盖 + 下压", "关键接触关系", GOLD),
        ("旋转 ↔ 重抓", "重复任务结构", GOLD),
        ("释放判断", "完成条件", PINK),
        ("提盖 + 放置", "后续目标", GREEN),
    ]
    for i, (title, desc, node_color) in enumerate(nodes):
        y = 190 + i * 80
        reached = i <= active
        draw.ellipse((940, y - 9, 958, y + 9), fill=node_color if reached else GRID)
        if i < len(nodes) - 1:
            draw.line((949, y + 12, 949, y + 66), fill=node_color if i < active else GRID, width=3)
        draw_text(draw, (978, y - 14), title, 21, WHITE if reached else MUTED, medium=True)
        draw_text(draw, (978, y + 18), desc, 16, node_color if reached else MUTED)
    draw.rounded_rectangle((932, 556, 1216, 600), radius=12, fill=(22, 31, 45), outline=CYAN)
    draw_text(draw, (1074, 578), "NO TRAJECTORY REPLAY", 16, CYAN, mono=True, anchor="mm")
    subtitle(image, "示范只提供意图、阶段顺序与关键接触关系；机器人不会逐帧复刻手轨迹。", CYAN)
    return image


def render_decomposition(t: float, duration: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    top_chrome(image, "02  SKILL COMPILATION", GOLD, t / duration)
    draw = ImageDraw.Draw(image)
    draw_text(draw, (640, 112), "同一段示范，被编译成三种控制权限", 35, WHITE, black=True, anchor="mm")
    cards = [
        (70, "AGENTIC", "自由空间靠近 / 放置", "agent 选目标，MoveIt 规划路径", CYAN),
        (455, "RELATIVE POSE", "抓盖 / 下压 / 绕轴旋转 / 提盖", "动作约束在 live bottle frame", GOLD),
        (840, "TACTILE", "接触 / 下压 / 重抓 / 释放", "只决定阶段是否继续与完成", PINK),
    ]
    reveal = clamp(t / (duration * 0.70))
    for i, (x, title, action, desc, color) in enumerate(cards):
        local = smooth((reveal * 3.0 - i) / 1.0)
        y = int(205 + (1.0 - local) * 35)
        rounded_panel(image, (x, y, x + 340, y + 286), fill=PANEL, outline=color, radius=24, width=2)
        draw.ellipse((x + 28, y + 28, x + 68, y + 68), fill=tuple(int(c * 0.25) for c in color), outline=color, width=3)
        draw_text(draw, (x + 86, y + 48), title, 21, color, black=True, mono=True, anchor="lm")
        draw_text(draw, (x + 28, y + 112), action, 23, WHITE, medium=True)
        draw.line((x + 28, y + 154, x + 312, y + 154), fill=GRID, width=2)
        for j, line in enumerate(desc.split("，")):
            draw_text(draw, (x + 28, y + 190 + j * 36), line, 18, MUTED)
    # Animated connectors make the authority hand-off explicit.
    for x0, x1, color in [(410, 455, GOLD), (795, 840, PINK)]:
        draw.line((x0, 348, x1, 348), fill=GRID, width=4)
        end = x0 + int((x1 - x0) * smooth((t - 2.0) / 1.5))
        draw.line((x0, 348, end, 348), fill=color, width=4)
        if end > x0 + 8:
            draw.polygon([(end, 348), (end - 11, 341), (end - 11, 355)], fill=color)
    draw_text(draw, (640, 550), "自由空间可以灵活规划；一旦接触物体，就切换到可解释的任务坐标与触觉条件。", 22, (203, 216, 229), anchor="mm")
    subtitle(image, "agent 负责“怎么到达”，相对位姿负责“怎么交互”，触觉负责“何时切换”。", GOLD)
    return image


def render_localization(t: float, duration: float, localization: Image.Image, cap_mask: Image.Image) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    paste_layer(image, contain(localization, (42, 90, 730, 610), radius=22))
    top_chrome(image, "03  LIVE RGB-D GROUNDING", CYAN, t / duration)
    draw = ImageDraw.Draw(image)
    pill(draw, (64, 108), "D435 · ALIGNED DEPTH", CYAN, size=17)
    # Mask inset is real evidence from the localization run.
    inset = cap_mask.convert("RGB").resize((160, 120), Image.Resampling.NEAREST)
    image.paste(inset, (535, 455))
    draw.rounded_rectangle((532, 452, 698, 578), radius=10, outline=GOLD, width=2)
    draw_text(draw, (615, 590), "cap mask", 14, GOLD, mono=True, anchor="mm")
    rounded_panel(image, (770, 90, 1238, 610), fill=PANEL, outline=GRID)
    draw_text(draw, (802, 126), "建立实时瓶盖任务坐标", 28, WHITE, black=True)
    steps = [
        ("01", "黑色瓶盖分割", "显式 ROI 内的主连通域"),
        ("02", "RGB-D 反投影", "2,918 个有效三维点"),
        ("03", "瓶轴 +Z", "当前直立瓶场景"),
        ("04", "robot_aligned", "对齐到 FR3 base frame"),
    ]
    for i, (num, title, desc) in enumerate(steps):
        y = 194 + i * 73
        active = smooth((t - i * 0.55) / 0.75)
        c = GOLD if i == 3 else CYAN
        draw_text(draw, (806, y), num, 18, c if active > 0.3 else GRID, mono=True)
        draw_text(draw, (852, y - 6), title, 20, WHITE if active > 0.3 else MUTED, medium=True)
        draw_text(draw, (852, y + 24), desc, 16, MUTED)
    draw.rounded_rectangle((798, 493, 1210, 572), radius=14, fill=(9, 17, 28), outline=GOLD, width=2)
    draw_text(draw, (818, 515), "EXECUTION ANCHOR / m", 14, GOLD, mono=True)
    draw_text(draw, (818, 552), "[0.57084, 0.06114, 0.08011]", 21, WHITE, mono=True)
    metric_card(image, (770, 620 - 1, 915, 708 - 1), "99.45%", "有效深度", CYAN)
    metric_card(image, (931, 619, 1076, 707), "43.6", "直径 / mm", GOLD)
    metric_card(image, (1092, 619, 1237, 707), "+Z", "瓶轴", GREEN)
    # The localization scene uses its own bottom metrics, so no regular subtitle.
    return image


def approach_source_time(local_t: float, duration: float) -> float:
    # Three pieces preserve the visually meaningful close while compressing waits.
    if local_t < 4.0:
        return 6.714 + (15.746 - 6.714) * (local_t / 4.0)
    if local_t < 5.5:
        return 15.746 + (16.812 - 15.746) * ((local_t - 4.0) / 1.5)
    return 16.812 + (29.013 - 16.812) * ((local_t - 5.5) / (duration - 5.5))


def render_approach(t: float, duration: float, reader: VideoReader) -> Image.Image:
    source_t = approach_source_time(t, duration)
    frame = reader.frame_at(source_t)
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    paste_layer(image, contain(frame, (42, 86, 866, 624), radius=22))
    top_chrome(image, "04  PLAN → CONTACT", CYAN if source_t < 11.1 else GOLD, t / duration)
    draw = ImageDraw.Draw(image)
    rounded_panel(image, (900, 90, 1238, 624), fill=PANEL, outline=GRID)
    if source_t < 11.1:
        stage, color, title, lines = "AGENTIC", CYAN, "自由空间靠近", ["目标：瓶盖上方 25 mm", "MoveIt 规划无碰路径"]
    elif source_t < 15.75:
        stage, color, title, lines = "RELATIVE POSE", GOLD, "沿瓶轴下探", ["保持 live cap frame", "到达接触高度"]
    elif source_t < 16.82:
        stage, color, title, lines = "RELATIVE POSE", GOLD, "Hand-E 抓住瓶盖", ["开口目标 18 mm", "保留实际触觉接触"]
    else:
        stage, color, title, lines = "TACTILE", PINK, "任务现场自标定", ["无载 → 夹盖", "已知 2 mm 下压"]
    pill(draw, (927, 120), stage, color, size=17)
    draw_text(draw, (927, 195), title, 28, WHITE, black=True)
    for i, line in enumerate(lines):
        draw.ellipse((929, 260 + i * 48, 941, 272 + i * 48), fill=color)
        draw_text(draw, (960, 266 + i * 48), line, 19, MUTED, anchor="lm")
    draw.line((934, 390, 1205, 390), fill=GRID, width=2)
    draw_text(draw, (934, 425), "控制权切换", 17, MUTED, medium=True)
    labels = [("PLAN", CYAN), ("DESCEND", GOLD), ("PRESS", PINK), ("TWIST", GOLD)]
    current = 0 if source_t < 11.1 else 1 if source_t < 16.82 else 2
    for i, (label, c) in enumerate(labels):
        x = 934 + i * 71
        draw.ellipse((x, 474, x + 18, 492), fill=c if i <= current else GRID)
        if i < len(labels) - 1:
            draw.line((x + 20, 483, x + 67, 483), fill=c if i < current else GRID, width=3)
        draw_text(draw, (x + 9, 518), label, 11, c if i <= current else MUTED, mono=True, anchor="mm")
    draw.rounded_rectangle((926, 553, 1212, 594), radius=11, fill=(20, 28, 41), outline=color)
    draw_text(draw, (1069, 573), f"source t = {source_t:05.1f} s", 14, color, mono=True, anchor="mm")
    subtitle(image, "定位给目标，规划给路径；进入接触后，触觉决定何时允许开始旋转。", color)
    return image


TWIST_INTERVALS = [
    (29.049, 37.094),
    (66.791, 74.694),
    (104.470, 113.750),
    (145.909, 152.930),
    (179.253, 185.807),
    (212.406, 219.340),
]


def loop_phase(source_t: float) -> tuple[int, str, tuple[int, int, int]]:
    for i, (start, end) in enumerate(TWIST_INTERVALS):
        if start <= source_t <= end:
            return i, "TWIST 90°", GOLD
        if i < len(TWIST_INTERVALS) - 1 and end < source_t < TWIST_INTERVALS[i + 1][0]:
            return i, "UNLOAD → RESET → REGRASP", CYAN
    return 0, "PRESS", PINK


def render_loop(t: float, duration: float, reader: VideoReader) -> Image.Image:
    source_t = 29.049 + (220.681 - 29.049) * clamp(t / duration)
    frame = reader.frame_at(source_t)
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    paste_layer(image, contain(frame, (470, 84, 1238, 624), radius=22))
    segment, phase, color = loop_phase(source_t)
    top_chrome(image, "05  TACTILE-SUPERVISED LOOP", PINK, t / duration)
    draw = ImageDraw.Draw(image)
    rounded_panel(image, (42, 90, 430, 624), fill=PANEL, outline=GRID)
    pill(draw, (68, 116), phase, color, size=16)
    draw_text(draw, (68, 184), "一个闭环，重复到释放", 28, WHITE, black=True)
    cycle = [
        ("PRESS", PINK),
        ("TWIST 90°", GOLD),
        ("UNLOAD", CYAN),
        ("RESET", CYAN),
        ("REGRASP", PINK),
    ]
    for i, (name, c) in enumerate(cycle):
        y = 246 + i * 50
        highlighted = ("TWIST" in phase and i == 1) or ("UNLOAD" in phase and i >= 2)
        draw.ellipse((74, y - 8, 90, y + 8), fill=c if highlighted else tuple(int(v * 0.45) for v in c))
        draw_text(draw, (108, y), name, 18, WHITE if highlighted else MUTED, mono=True, anchor="lm")
        if i < len(cycle) - 1:
            draw.line((82, y + 11, 82, y + 39), fill=GRID, width=2)
    draw_text(draw, (68, 516), "旋转段", 15, MUTED)
    for i in range(6):
        x = 68 + i * 51
        fill = GREEN if i < segment else color if i == segment else (29, 42, 58)
        draw.rounded_rectangle((x, 547, x + 39, 579), radius=9, fill=fill)
        draw_text(draw, (x + 19, 563), str(i + 1), 14, BG if i <= segment else MUTED, black=True, anchor="mm")
    # Task-axis helix: first revolution down, then up with the thread.
    axis_x = 399
    draw.line((axis_x, 239, axis_x, 468), fill=GREEN, width=3)
    draw.polygon([(axis_x, 228), (axis_x - 7, 242), (axis_x + 7, 242)], fill=GREEN)
    pts = []
    for j in range(80):
        q = j / 79
        y = 252 + int(185 * q)
        x = axis_x - 24 + int(18 * math.sin(q * math.pi * 6))
        pts.append((x, y))
    draw.line(pts, fill=GOLD, width=3)
    draw_text(draw, (360, 489), "bottle +Z", 12, GREEN, mono=True)
    subtitle(image, "旋转角来自任务几何；触觉监督接触，腕部行程决定何时自动回腕重抓。", color)
    return image


def plot_tactile(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], reveal: float) -> None:
    x0, y0, x1, y1 = box
    # Plot evidence from the six twist endpoints plus the two fresh and one
    # stationary final windows.
    x_values = [90, 180, 270, 360, 450, 540, 550, 560]
    y_values = [0.398425, 0.804700, 0.357721, 0.341253, 1.102016, 0.077348, 0.115201, 0.056725]
    colors = [PINK] * 5 + [GREEN, GREEN, GREEN]
    ymax = 1.20
    def point(x: float, y: float) -> tuple[int, int]:
        return (
            int(x0 + (x / 570.0) * (x1 - x0)),
            int(y1 - (y / ymax) * (y1 - y0)),
        )
    for frac in (0.0, 0.25, 0.50, 0.75, 1.0):
        yy = y1 - int(frac * (y1 - y0))
        draw.line((x0, yy, x1, yy), fill=(31, 45, 61), width=1)
        draw_text(draw, (x0 - 10, yy), f"{frac * ymax:.1f}", 12, MUTED, mono=True, anchor="rm")
    threshold = 0.3857055163140876
    threshold_y = point(0, threshold)[1]
    draw.line((x0, threshold_y, x1, threshold_y), fill=GOLD, width=2)
    draw_text(draw, (x1 - 4, threshold_y - 8), "35% peak = 0.386", 13, GOLD, mono=True, anchor="rb")
    count_float = clamp(reveal) * len(x_values)
    count = min(len(x_values), int(math.ceil(count_float)))
    pts = [point(x_values[i], y_values[i]) for i in range(count)]
    if len(pts) >= 2:
        draw.line(pts, fill=PINK, width=4, joint="curve")
    for i, p in enumerate(pts):
        r = 7 if i >= 5 else 5
        draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=colors[i], outline=WHITE, width=1)
        if i == 4:
            draw_text(draw, (p[0], p[1] - 20), "PEAK 1.102", 13, PINK, mono=True, anchor="mb")
        if i >= 5:
            draw_text(draw, (p[0], p[1] - 16), f"{y_values[i]:.3f}", 12, GREEN, mono=True, anchor="mb")
    draw_text(draw, ((x0 + x1) / 2, y1 + 32), "累计旋转角 / degree", 14, MUTED, anchor="mm")


def render_tactile(t: float, duration: float, reader: VideoReader) -> Image.Image:
    source_t = 212.356 + (220.681 - 212.356) * clamp(t / duration)
    frame = reader.frame_at(source_t)
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    paste_layer(image, contain(frame, (42, 100, 586, 604), radius=22))
    top_chrome(image, "06  RELEASE PREDICATE", PINK, t / duration)
    draw = ImageDraw.Draw(image)
    pill(draw, (64, 118), "RELATIVE RAW-Fz PROXY", PINK, size=15)
    rounded_panel(image, (620, 90, 1238, 604), fill=PANEL, outline=GRID)
    draw_text(draw, (650, 122), "触觉不是“看一眼”，而是状态转移条件", 25, WHITE, black=True)
    # Reveal historical points quickly, then the three final low windows.
    if t < 2.2:
        reveal = 5.0 / 8.0 * smooth(t / 2.2)
    else:
        reveal = 5.0 / 8.0 + 3.0 / 8.0 * smooth((t - 2.2) / 4.2)
    plot_tactile(draw, (680, 210, 1190, 456), reveal)
    draw_text(draw, (651, 178), "阻力代理 / raw sensor units", 14, MUTED)
    if t < 3.0:
        state_text, state_color = "记录上升阶段峰值", PINK
    elif t < 6.2:
        state_text, state_color = "连续低于释放阈值", GOLD
    else:
        state_text, state_color = "RELEASE VERIFIED", GREEN
    pill(draw, (650, 503), state_text, state_color, size=16)
    if t >= 6.2:
        draw_text(draw, (650, 566), "contact  0.318  >  0.0157", 18, GREEN, mono=True)
        draw_text(draw, (1010, 566), "≠ 掉盖", 17, WHITE, medium=True)
    if t < 3.0:
        caption = "超过一圈后才允许判断释放：先记住本轮上升阶段的阻力峰值。"
    elif t < 6.2:
        caption = "0.077、0.115 连续低于峰值的 35%，形成螺纹释放候选。"
    else:
        caption = "静止窗口仍为 0.057，且触觉接触还在：螺纹已脱离，不是掉盖。"
    subtitle(image, caption, state_color)
    return image


def render_final(t: float, duration: float, reader: VideoReader) -> Image.Image:
    source_t = 220.681 + (227.6 - 220.681) * clamp(t / duration)
    frame = reader.frame_at(source_t)
    image = cover(frame, (0, 0, WIDTH, HEIGHT), radius=0)
    dim(image, 52)
    draw = ImageDraw.Draw(image)
    top_chrome(image, "07  EXECUTION COMPLETE", GREEN, t / duration)
    pill(draw, (52, 94), "RELEASE_VERIFIED  →  LIFT", GREEN, size=18)
    draw_text(draw, (62, 168), "只在释放通过后提盖", 39, WHITE, black=True)
    draw_text(draw, (64, 224), "沿实时瓶轴  +30 mm", 26, GREEN, medium=True)
    # Animated vertical lift arrow.
    arrow = smooth(t / 2.0)
    x = 1137
    y_bottom = 555
    y_top = int(y_bottom - 210 * arrow)
    draw.line((x, y_bottom, x, y_top + 13), fill=GREEN, width=8)
    draw.polygon([(x, y_top), (x - 14, y_top + 24), (x + 14, y_top + 24)], fill=GREEN)
    draw_text(draw, (x - 28, (y_bottom + y_top) / 2), "+30 mm", 19, GREEN, mono=True, anchor="rm")
    metric_card(image, (60, 430, 306, 528), "540°", "任务轴旋转", GOLD)
    metric_card(image, (324, 430, 570, 528), "5×", "自动回腕重抓", CYAN)
    metric_card(image, (588, 430, 834, 528), "1.45 mm", "最终提盖误差", GREEN)
    draw.rounded_rectangle((62, 562, 835, 616), radius=16, fill=(7, 14, 24), outline=GREEN, width=2)
    draw_text(draw, (448, 589), "ONE DEMO  ·  OBJECT-CENTRIC  ·  TACTILE CLOSED LOOP", 17, WHITE, mono=True, anchor="mm")
    subtitle(image, "最终结果：相对位姿保证交互几何，触觉闭环决定完成时机。", GREEN)
    return image


def action_log_evidence(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload["events"]
    release = next(e for e in events if e.get("operation") == "release_verify")
    return {
        "release_predicate": release["predicate"],
        "peak": release["upfeed_global_peak_resistance_proxy_sensor_units"],
        "limit": release["maximum_release_proxy_sensor_units"],
        "final_stationary": release["after_proxy_sensor_units"],
        "contact_presence": release["after_presence_sensor_units"],
        "contact_minimum": release["minimum_presence_sensor_units"],
    }


def build_timeline(output: Path, sources: Sources) -> dict:
    timeline = {
        "schema": "demo2gripper.framework_explainer_timeline.v1",
        "output": str(output),
        "video": {"width": WIDTH, "height": HEIGHT, "fps": FPS, "duration_s": 66.0},
        "truth_labels": {
            "localization": "RGB-D cap frame with robot_aligned execution anchor",
            "tactile": "two GF220 FORCE6D windows logged; decisions use session-relative raw SDK Fz proxy",
            "execution": "run16: 540 degrees, five regrasps, release_verified, 30 mm lift",
            "limitations": [
                "human demo starts with bottle already upright",
                "run16 ends after LIFT_CAP and does not execute PLACE_CAP_ASIDE",
                "run16 did not save synchronized tactile RGB video",
                "raw Fz proxy is not labelled in newtons and is not a calibrated task-frame torque",
            ],
        },
        "evidence": action_log_evidence(sources.action_log),
        "scenes": [
            {"name": "hook", "output_s": [0.0, 4.0], "robot_source_s": [219.35, 227.4]},
            {"name": "human_demo", "output_s": [4.0, 12.5], "demo_source_s": [0.60, 10.333]},
            {"name": "skill_compilation", "output_s": [12.5, 19.5]},
            {"name": "rgbd_grounding", "output_s": [19.5, 27.0]},
            {"name": "plan_to_contact", "output_s": [27.0, 36.0], "robot_source_s": [6.714, 29.013]},
            {"name": "twist_regrasp_loop", "output_s": [36.0, 50.0], "robot_source_s": [29.049, 220.681]},
            {"name": "release_predicate", "output_s": [50.0, 59.0], "robot_source_s": [212.356, 220.681]},
            {"name": "lift_result", "output_s": [59.0, 66.0], "robot_source_s": [220.681, 227.6]},
        ],
    }
    return timeline


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    default_dir = repo / "runs/live/framework_explainers_20261023"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=default_dir / "medicine_bottle_framework_explainer.mp4")
    parser.add_argument("--timeline", type=Path, default=default_dir / "medicine_bottle_framework_timeline.json")
    parser.add_argument("--preview-frame", type=Path, default=default_dir / "medicine_bottle_framework_preview.jpg")
    parser.add_argument("--crf", type=int, default=18)
    return parser.parse_args()


def render(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    sources = Sources(
        demo=repo / "runs/live/medicine_bottle_demo_20261023_01/analysis/demo_trim_1110_1274.mp4",
        robot=repo / "runs/live/medicine_bottle_robot_video_20261023_01/robot_execution_front_full.mp4",
        localization=repo / "runs/live/medicine_bottle_execution_20261023_01/localization_execute/diagnostic_overlay.png",
        cap_mask=repo / "runs/live/medicine_bottle_execution_20261023_01/localization_execute/cap_mask.png",
        action_log=repo / "runs/live/medicine_bottle_execution_20261023_16_video_fast/action_log.json",
    )
    for path in sources.__dict__.values():
        if not path.exists():
            raise FileNotFoundError(path)
    evidence = action_log_evidence(sources.action_log)
    expected = {
        "peak": 1.1020157608973933,
        "limit": 0.3857055163140876,
        "final_stationary": 0.05672473664418365,
    }
    for key, value in expected.items():
        if not math.isclose(float(evidence[key]), value, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(f"unexpected run16 {key}: {evidence[key]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    timeline = build_timeline(args.output.resolve(), sources)
    args.timeline.write_text(json.dumps(timeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    localization = Image.open(sources.localization).convert("RGB")
    cap_mask = Image.open(sources.cap_mask).convert("L")
    readers = {
        "hook": VideoReader(sources.robot, 219.35),
        "demo": VideoReader(sources.demo, 0.60),
        "approach": VideoReader(sources.robot, 6.714),
        "loop": VideoReader(sources.robot, 29.049),
        "tactile": VideoReader(sources.robot, 212.356),
        "final": VideoReader(sources.robot, 220.681),
    }
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    preview: Image.Image | None = None
    total_frames = int(66.0 * FPS)
    try:
        for index in range(total_frames):
            global_t = index / FPS
            if global_t < 4.0:
                frame = render_hook(global_t, 4.0, readers["hook"])
            elif global_t < 12.5:
                frame = render_demo(global_t - 4.0, 8.5, readers["demo"])
            elif global_t < 19.5:
                frame = render_decomposition(global_t - 12.5, 7.0)
            elif global_t < 27.0:
                frame = render_localization(global_t - 19.5, 7.5, localization, cap_mask)
            elif global_t < 36.0:
                frame = render_approach(global_t - 27.0, 9.0, readers["approach"])
            elif global_t < 50.0:
                frame = render_loop(global_t - 36.0, 14.0, readers["loop"])
            elif global_t < 59.0:
                frame = render_tactile(global_t - 50.0, 9.0, readers["tactile"])
            else:
                frame = render_final(global_t - 59.0, 7.0, readers["final"])
            if index == int(56.8 * FPS):
                preview = frame.copy()
            process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with code {return_code}")
    finally:
        for reader in readers.values():
            reader.close()
        if process.poll() is None:
            process.kill()
    if preview is not None:
        preview.save(args.preview_frame, quality=94)


def main() -> int:
    args = parse_args()
    render(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
