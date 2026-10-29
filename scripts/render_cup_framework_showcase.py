#!/usr/bin/env python3
"""Render the paper-cup framework explainer from the preserved local evidence.

The movie intentionally distinguishes the framework's role-aware tactile policy
from Run 08's shadow-only tactile diagnostic and operator visual confirmation.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "runs/live/framework_showcase_20261023/cup"
OUT_MP4 = OUT_DIR / "cup_framework_explainer_720p.mp4"
OUT_TIMELINE = OUT_DIR / "cup_framework_explainer_timeline.json"
OUT_SCRIPT = OUT_DIR / "cup_framework_explainer_script.md"
OUT_SRT = OUT_DIR / "cup_framework_explainer.zh-CN.srt"

HUMAN_STAGE = (
    REPO
    / "runs/live/cup_top_demo_20261023_01/analysis_image_stages/"
    "front_rgb_stage_preview.mp4"
)
PERCEPTION = (
    REPO
    / "runs/live/cup_top_demo_20261023_01/diagnostics/sam2_rtmpose_rgbd.mp4"
)
RIM_RELATION = (
    REPO
    / "runs/live/cup_top_demo_20261023_01/cup_rim_relation_review_v2/preview.png"
)
ROBOT_RUN = (
    REPO
    / "runs/live/cup_tactile_demo_20261023_08/rendered/"
    "cup_tactile_demo_run08_single_cup_visual_confirmed_final.mp4"
)

WIDTH = 1280
HEIGHT = 720
FPS = 30

BLACK = (5, 8, 13)
WHITE = (239, 244, 248)
MUTED = (147, 165, 176)
CYAN = (51, 221, 255)
GOLD = (255, 196, 70)
PINK = (255, 83, 177)
GREEN = (77, 232, 146)
RED = (255, 82, 92)
BLUE = (88, 145, 255)

FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


@lru_cache(maxsize=64)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def alpha(color: tuple[int, int, int], value: int) -> tuple[int, int, int, int]:
    return color + (value,)


def ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def draw_text(
    image: Image.Image,
    xy: tuple[float, float],
    text: str,
    size: int,
    color: tuple[int, int, int] = WHITE,
    *,
    bold: bool = False,
    anchor: str = "la",
    stroke: int = 0,
) -> None:
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=font(size, bold),
        fill=color,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=BLACK,
    )


def rounded_rect(
    image: Image.Image,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int] | tuple[int, int, int, int],
    *,
    radius: int = 18,
    outline: tuple[int, int, int] | None = None,
    width: int = 1,
) -> None:
    ImageDraw.Draw(image).rounded_rectangle(
        box, radius=radius, fill=fill, outline=outline, width=width
    )


def base_frame(scene_index: int, t: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
    draw = ImageDraw.Draw(image)
    # Sparse scientific grid.
    for x in range(0, WIDTH + 1, 64):
        draw.line((x, 0, x, HEIGHT), fill=(10, 19, 27), width=1)
    for y in range(0, HEIGHT + 1, 64):
        draw.line((0, y, WIDTH, y), fill=(10, 19, 27), width=1)
    rng = np.random.default_rng(20261023 + scene_index)
    for px, py, radius in zip(
        rng.integers(0, WIDTH, 42),
        rng.integers(0, HEIGHT, 42),
        rng.integers(1, 3, 42),
    ):
        pulse = 70 + int(35 * math.sin(t * 1.2 + px * 0.02))
        draw.ellipse(
            (int(px - radius), int(py - radius), int(px + radius), int(py + radius)),
            fill=(20, min(110, pulse), min(130, pulse + 10)),
        )
    return image


def paste_fit(
    canvas: Image.Image,
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    crop_fill: bool = False,
    radius: int = 18,
) -> None:
    x0, y0, x1, y1 = box
    target_w, target_h = x1 - x0, y1 - y0
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    source = Image.fromarray(rgb)
    sw, sh = source.size
    scale = max(target_w / sw, target_h / sh) if crop_fill else min(target_w / sw, target_h / sh)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    source = source.resize((nw, nh), Image.Resampling.LANCZOS)
    if crop_fill:
        left = max(0, (nw - target_w) // 2)
        top = max(0, (nh - target_h) // 2)
        source = source.crop((left, top, left + target_w, top + target_h))
    else:
        backing = Image.new("RGB", (target_w, target_h), (3, 5, 8))
        backing.paste(source, ((target_w - nw) // 2, (target_h - nh) // 2))
        source = backing
    mask = Image.new("L", (target_w, target_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, target_w, target_h), radius=radius, fill=255)
    canvas.paste(source, (x0, y0), mask)


class Clip:
    def __init__(self, path: Path, start_s: float, speed: float = 1.0) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.source_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 15.0)
        self.start_frame = int(round(start_s * self.source_fps))
        self.speed = speed
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.current_index = self.start_frame - 1
        self.current: np.ndarray | None = None

    def frame(self, local_s: float) -> np.ndarray:
        target = self.start_frame + int(local_s * self.speed * self.source_fps)
        if target < self.current_index:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.current_index = target - 1
            self.current = None
        while self.current_index < target:
            ok, frame = self.capture.read()
            if not ok:
                break
            self.current = frame
            self.current_index += 1
        if self.current is None:
            raise RuntimeError(f"no frame available in {self.path} at {local_s:.3f}s")
        return self.current

    def close(self) -> None:
        self.capture.release()


@dataclass
class Scene:
    key: str
    duration_s: float
    subtitle: str
    evidence: str
    render: Callable[[float, int], Image.Image]


def header(image: Image.Image, label: str, color: tuple[int, int, int]) -> None:
    draw_text(image, (38, 27), "TACTILE AGENT / CASE 01", 18, MUTED, bold=True)
    rounded_rect(image, (1000, 18, 1240, 51), (12, 26, 33), radius=16, outline=color)
    draw_text(image, (1120, 35), label, 16, WHITE, bold=True, anchor="mm")


def subtitle_bar(image: Image.Image, text: str, color: tuple[int, int, int] = WHITE) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    rounded_rect(overlay, (80, 636, 1200, 701), (2, 6, 10, 224), radius=20, outline=(36, 55, 65))
    image.alpha_composite(overlay) if image.mode == "RGBA" else image.paste(overlay, (0, 0), overlay)
    draw_text(image, (640, 669), text, 28, color, bold=True, anchor="mm")


def small_note(image: Image.Image, text: str, color: tuple[int, int, int] = MUTED) -> None:
    draw_text(image, (1240, 612), text, 16, color, anchor="ra")


def make_hook() -> Scene:
    clip = Clip(ROBOT_RUN, 28.6, 1.7)

    def render(t: float, index: int) -> Image.Image:
        image = base_frame(index, t)
        paste_fit(image, clip.frame(t), (20, 18, 1260, 625), crop_fill=False, radius=22)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rectangle((0, 0, WIDTH, 145), fill=(0, 0, 0, 158))
        image.paste(overlay, (0, 0), overlay)
        draw_text(image, (54, 54), "只取最上面一只", 54, WHITE, bold=True)
        draw_text(image, (56, 111), "一次示范 · 关键位姿迁移 · 触觉监工", 25, GREEN, bold=True)
        subtitle_bar(image, "先看结果：FR3 成功抬起一只纸杯", GREEN)
        return image

    return Scene("hook", 4.0, "先看结果：FR3 成功抬起一只纸杯", "Run08 front RGB", render)


def make_title() -> Scene:
    def render(t: float, index: int) -> Image.Image:
        image = base_frame(index, t)
        header(image, "FRAMEWORK", CYAN)
        p = ease(t / 1.0)
        draw = ImageDraw.Draw(image)
        draw.line((120, 350, 1160 * p, 350), fill=CYAN, width=4)
        points = [(160, "一次示范"), (440, "阶段意图"), (720, "关键几何"), (1000, "真机闭环")]
        colors = [WHITE, CYAN, GOLD, PINK]
        for i, ((x, label), color) in enumerate(zip(points, colors)):
            q = ease((t - i * 0.35) / 0.7)
            radius = int(7 + 14 * q)
            draw.ellipse((x - radius, 350 - radius, x + radius, 350 + radius), fill=color)
            if q > 0.05:
                draw_text(image, (x, 295), label, 25, color, bold=True, anchor="mm")
        draw_text(image, (640, 160), "不是轨迹复读", 48, WHITE, bold=True, anchor="mm")
        draw_text(image, (640, 225), "而是把演示编译成可执行技能", 42, CYAN, bold=True, anchor="mm")
        subtitle_bar(image, "自由空间重新规划，只迁移物体交互的关键关系")
        return image

    return Scene(
        "principle_title",
        4.0,
        "自由空间重新规划，只迁移物体交互的关键关系",
        "framework contract",
        render,
    )


def make_human_demo() -> Scene:
    clip = Clip(HUMAN_STAGE, 8.7, 1.35)
    stages = ["APPROACH", "PRESHAPE", "GRASP", "LIFT", "TRANSPORT", "PLACE", "RELEASE", "RETREAT"]

    def render(t: float, index: int) -> Image.Image:
        image = base_frame(index, t)
        header(image, "ONE-SHOT DEMO", WHITE)
        paste_fit(image, clip.frame(t), (44, 76, 876, 612), crop_fill=False, radius=22)
        draw_text(image, (922, 104), "人类只示范一次", 31, WHITE, bold=True)
        draw_text(image, (922, 145), "系统读取意图与交互几何", 20, MUTED)
        for i, name in enumerate(stages):
            y = 194 + i * 48
            active = min(len(stages) - 1, int(t / max(0.1, 7.4 / len(stages))))
            color = GREEN if i < active else (GOLD if i == active else (62, 76, 86))
            ImageDraw.Draw(image).ellipse((925, y - 6, 937, y + 6), fill=color)
            draw_text(image, (953, y), name, 18, color if i <= active else MUTED, bold=i == active, anchor="lm")
        subtitle_bar(image, "视频被自动拆成靠近、抓取、抬升、搬运、放置与撤离")
        return image

    return Scene(
        "human_demo",
        7.4,
        "视频被自动拆成靠近、抓取、抬升、搬运、放置与撤离",
        "front RGB semantic stages, 8.87–17.81 s",
        render,
    )


def make_perception() -> Scene:
    clip = Clip(PERCEPTION, 8.85, 1.05)

    def render(t: float, index: int) -> Image.Image:
        image = base_frame(index, t)
        header(image, "PERCEPTION", CYAN)
        paste_fit(image, clip.frame(t), (54, 80, 930, 612), crop_fill=False, radius=22)
        cards = [
            ("SAM2", "连续分割手与目标杯", CYAN),
            ("RTMPose Hand5", "恢复 21 个手部关键点", GOLD),
            ("Aligned RGB-D", "建立三维杯沿 task frame", GREEN),
        ]
        for i, (title, line, color) in enumerate(cards):
            y0 = 106 + i * 146
            rounded_rect(image, (965, y0, 1235, y0 + 112), (11, 18, 25), radius=18, outline=color, width=2)
            draw_text(image, (990, y0 + 29), title, 21, color, bold=True)
            draw_text(image, (990, y0 + 70), line, 16, WHITE)
        small_note(image, "362 frames · 14.99 Hz · aligned metric depth")
        subtitle_bar(image, "先回答“杯沿在哪里”，再理解“手是怎样夹的”")
        return image

    return Scene(
        "perception",
        6.2,
        "先回答“杯沿在哪里”，再理解“手是怎样夹的”",
        "SAM2 + RTMPose + aligned RGB-D",
        render,
    )


def make_relation() -> Scene:
    still = cv2.imread(str(RIM_RELATION))
    if still is None:
        raise RuntimeError(f"cannot open {RIM_RELATION}")

    def render(t: float, index: int) -> Image.Image:
        image = base_frame(index, t)
        header(image, "RELATIVE POSE", GOLD)
        # Slow push-in while retaining both evidence panels.
        h, w = still.shape[:2]
        zoom = 1.0 + 0.025 * ease(t / 5.6)
        crop_w, crop_h = int(w / zoom), int(h / zoom)
        x0, y0 = (w - crop_w) // 2, (h - crop_h) // 2
        crop = still[y0 : y0 + crop_h, x0 : x0 + crop_w]
        paste_fit(image, crop, (42, 84, 1238, 535), crop_fill=True, radius=22)
        rounded_rect(image, (78, 555, 365, 620), (15, 17, 19), radius=18, outline=GOLD)
        rounded_rect(image, (391, 555, 690, 620), (15, 17, 19), radius=18, outline=GOLD)
        rounded_rect(image, (716, 555, 1202, 620), (15, 17, 19), radius=18, outline=CYAN)
        draw_text(image, (221, 587), "径向偏移 30.4 mm", 18, GOLD, bold=True, anchor="mm")
        draw_text(image, (541, 587), "轴向偏移 23.6 mm", 18, GOLD, bold=True, anchor="mm")
        draw_text(image, (959, 587), "只迁移 T_rim→pinch，不回放手的路径", 18, CYAN, bold=True, anchor="mm")
        subtitle_bar(image, "演示贡献关键接触位姿；自由运动交给 agent")
        return image

    return Scene(
        "relative_pose",
        5.6,
        "演示贡献关键接触位姿；自由运动交给 agent",
        "cup_rim_relation.v1",
        render,
    )


def make_stage_graph() -> Scene:
    rows = [
        (["OBSERVE", "GROUND", "APPROACH"], CYAN, "在线感知 / agent + MoveIt"),
        (["RIM ALIGN"], GOLD, "迁移人手—杯沿相对位姿"),
        (["PINCH", "SEPARATE", "VERIFY"], PINK, "任务条件化触觉监工"),
        (["CLEAR LIFT", "TRANSPORT", "PLACE", "RELEASE", "RETREAT"], CYAN, "按意图重新规划"),
    ]

    def render(t: float, index: int) -> Image.Image:
        image = base_frame(index, t)
        header(image, "SKILL COMPILER", CYAN)
        draw_text(image, (64, 91), "一次演示 → 12 个可审计阶段", 38, WHITE, bold=True)
        draw_text(image, (66, 138), "每个阶段都有 authority、完成条件和失败转移", 20, MUTED)
        y = 214
        delay = 0.0
        for names, color, explanation in rows:
            x = 72
            draw_text(image, (70, y - 30), explanation, 16, color, bold=True)
            for name in names:
                node_w = max(134, 25 + len(name) * 12)
                q = ease((t - delay) / 0.55)
                if q > 0.01:
                    rounded_rect(image, (x, y, x + node_w, y + 58), (12, 20, 27), radius=16, outline=color, width=2)
                    draw_text(image, (x + node_w / 2, y + 30), name, 17, color, bold=True, anchor="mm")
                x += node_w + 24
                delay += 0.12
            y += 102
        legend = [(CYAN, "AGENTIC"), (GOLD, "RELATIVE POSE"), (PINK, "TACTILE")]
        x = 805
        for color, label in legend:
            ImageDraw.Draw(image).ellipse((x, 147, x + 12, 159), fill=color)
            draw_text(image, (x + 22, 153), label, 14, color, bold=True, anchor="lm")
            x += 150
        subtitle_bar(image, "关键处模仿人；其余阶段根据当前场景重新规划")
        return image

    return Scene(
        "stage_graph",
        7.2,
        "关键处模仿人；其余阶段根据当前场景重新规划",
        "top_cup_stage_plan.v3",
        render,
    )


def make_tactile_policy() -> Scene:
    def render(t: float, index: int) -> Image.Image:
        image = base_frame(index, t)
        header(image, "TACTILE POLICY / DESIGN", PINK)
        draw_text(image, (58, 87), "触觉不规划轨迹，它只签发阶段许可", 37, WHITE, bold=True)
        sensors = [
            ("WARPED / DIFF", "接触位置 · 形状 · 多带", 82),
            ("FORCE6D", "法向低力带 · 提取剪切", 303),
            ("MARKER / SLIP", "耦合 · 初滑 · 屈曲", 524),
        ]
        for title, line, x in sensors:
            rounded_rect(image, (x, 169, x + 198, 288), (12, 17, 24), radius=20, outline=PINK, width=2)
            draw_text(image, (x + 99, 207), title, 19, PINK, bold=True, anchor="mm")
            draw_text(image, (x + 99, 252), line, 15, WHITE, anchor="mm")
        draw = ImageDraw.Draw(image)
        hub_x, hub_y = 922, 228
        for _, _, x in sensors:
            draw.line((x + 198, 228, hub_x - 70, hub_y), fill=PINK, width=3)
        draw.ellipse((hub_x - 66, hub_y - 66, hub_x + 66, hub_y + 66), fill=(36, 11, 29), outline=PINK, width=4)
        draw_text(image, (hub_x, hub_y - 10), "INNER", 19, PINK, bold=True, anchor="mm")
        draw_text(image, (hub_x, hub_y + 22), "+ OUTER", 19, PINK, bold=True, anchor="mm")
        rules = [
            (RED, "NO CONTACT", "微闭 / 下探"),
            (RED, "TOO DEEP", "卸载 → 上调"),
            (GOLD, "ROLE MISMATCH", "卸载 → 重对中"),
            (GREEN, "K× ROLE MATCH", "允许分离抬升"),
        ]
        for i, (color, state, action) in enumerate(rules):
            x = 76 + (i % 2) * 582
            y = 350 + (i // 2) * 105
            rounded_rect(image, (x, y, x + 540, y + 74), (11, 18, 25), radius=18, outline=color, width=2)
            draw_text(image, (x + 22, y + 26), state, 18, color, bold=True)
            draw_text(image, (x + 518, y + 39), action, 20, WHITE, bold=True, anchor="rm")
        draw_text(image, (640, 580), "两指角色可不对称 · 不用前视图像“数杯子”", 19, MUTED, anchor="mm")
        subtitle_bar(image, "接触判据决定继续、重试还是停止")
        return image

    return Scene(
        "tactile_policy",
        7.4,
        "接触判据决定继续、重试还是停止",
        "framework policy; not Run08 online control",
        render,
    )


def make_robot_probe() -> Scene:
    clip = Clip(ROBOT_RUN, 1.35, 2.35)

    def render(t: float, index: int) -> Image.Image:
        image = base_frame(index, t)
        paste_fit(image, clip.frame(t), (20, 17, 1260, 619), crop_fill=False, radius=22)
        rounded_rect(image, (44, 36, 419, 110), (32, 7, 12), radius=18, outline=RED, width=3)
        draw_text(image, (68, 63), "试探：TCP 低 6 mm", 27, RED, bold=True)
        draw_text(image, (69, 94), "close → NO LIFT → unload", 16, WHITE, bold=True)
        small_note(image, "Run08：该 RETRY 为脚本预设；触觉同步 shadow 记录", GOLD)
        subtitle_bar(image, "先做错误高度试探：闭合，但禁止抬升", RED)
        return image

    return Scene(
        "robot_probe",
        7.9,
        "先做错误高度试探：闭合，但禁止抬升",
        "Run08 scripted retry + tactile shadow",
        render,
    )


def make_robot_reacquire() -> Scene:
    clip = Clip(ROBOT_RUN, 12.0, 1.75)

    def render(t: float, index: int) -> Image.Image:
        image = base_frame(index, t)
        paste_fit(image, clip.frame(t), (20, 17, 1260, 619), crop_fill=False, radius=22)
        rounded_rect(image, (44, 36, 472, 110), (30, 22, 5), radius=18, outline=GOLD, width=3)
        draw_text(image, (68, 63), "卸载 → 上调 6 mm → 重夹", 26, GOLD, bold=True)
        draw_text(image, (69, 94), "reviewed TCP error = 0.168 mm", 16, WHITE, bold=True)
        small_note(image, "tactile: raw UVC + offline absdiff shadow", PINK)
        subtitle_bar(image, "恢复到示范迁移后的关键接触位姿", GOLD)
        return image

    return Scene(
        "robot_reacquire",
        7.0,
        "恢复到示范迁移后的关键接触位姿",
        "Run08 pose gate + scripted reacquisition",
        render,
    )


def make_robot_lift() -> Scene:
    clip = Clip(ROBOT_RUN, 24.0, 1.15)

    def render(t: float, index: int) -> Image.Image:
        image = base_frame(index, t)
        paste_fit(image, clip.frame(t), (20, 17, 1260, 619), crop_fill=False, radius=22)
        rounded_rect(image, (44, 36, 457, 110), (5, 31, 20), radius=18, outline=GREEN, width=3)
        draw_text(image, (68, 63), "PASS · 单杯抬升约 67 mm", 26, GREEN, bold=True)
        draw_text(image, (69, 94), "remaining stack stays below", 16, WHITE, bold=True)
        small_note(image, "最终单杯结果：front RGB / operator visual confirmation", GREEN)
        subtitle_bar(image, "目标杯随夹爪上升，其余杯堆留在原位", GREEN)
        return image

    return Scene(
        "robot_lift",
        9.9,
        "目标杯随夹爪上升，其余杯堆留在原位",
        "Run08 visual outcome frames 489 and 534",
        render,
    )


def make_outro() -> Scene:
    def render(t: float, index: int) -> Image.Image:
        image = base_frame(index, t)
        header(image, "WHAT THIS PROVES", GREEN)
        draw_text(image, (640, 105), "这个 framework 的三层分工", 40, WHITE, bold=True, anchor="mm")
        cards = [
            (CYAN, "AGENTIC", "按意图重规划自由运动"),
            (GOLD, "RELATIVE POSE", "只迁移接触关键几何"),
            (PINK, "TACTILE SUPERVISOR", "用任务条件判据守住阶段边界"),
        ]
        for i, (color, title, line) in enumerate(cards):
            x0 = 58 + i * 411
            q = ease((t - i * 0.32) / 0.8)
            y0 = int(210 + (1 - q) * 40)
            rounded_rect(image, (x0, y0, x0 + 370, y0 + 235), (10, 17, 24), radius=24, outline=color, width=3)
            draw_text(image, (x0 + 185, y0 + 57), title, 22, color, bold=True, anchor="mm")
            draw_text(image, (x0 + 185, y0 + 120), line, 20, WHITE, bold=True, anchor="mm")
            draw_text(image, (x0 + 185, y0 + 177), ["当前场景", "交互点", "阶段许可"][i], 18, MUTED, anchor="mm")
        draw_text(
            image,
            (640, 524),
            "Run08 证据边界：触觉为 shadow；最终单杯由前视确认；机器人片段止于 LIFT / HOLD",
            17,
            MUTED,
            anchor="mm",
        )
        subtitle_bar(image, "一次示范，生成可解释、可检查、可恢复的机器人技能")
        return image

    return Scene(
        "outro",
        6.0,
        "一次示范，生成可解释、可检查、可恢复的机器人技能",
        "evidence boundary stated on screen",
        render,
    )


def build_scenes() -> list[Scene]:
    return [
        make_hook(),
        make_title(),
        make_human_demo(),
        make_perception(),
        make_relation(),
        make_stage_graph(),
        make_tactile_policy(),
        make_robot_probe(),
        make_robot_reacquire(),
        make_robot_lift(),
        make_outro(),
    ]


def timestamp_srt(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_metadata(scenes: list[Scene]) -> None:
    timeline: list[dict[str, object]] = []
    srt_parts: list[str] = []
    cursor = 0.0
    for number, scene in enumerate(scenes, start=1):
        start = cursor
        end = start + scene.duration_s
        timeline.append(
            {
                "index": number,
                "scene": scene.key,
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "duration_s": scene.duration_s,
                "subtitle": scene.subtitle,
                "evidence": scene.evidence,
            }
        )
        srt_parts.extend(
            [
                str(number),
                f"{timestamp_srt(start)} --> {timestamp_srt(end)}",
                scene.subtitle,
                "",
            ]
        )
        cursor = end
    OUT_TIMELINE.write_text(
        json.dumps(
            {
                "schema": "demo2gripper.framework_showcase_timeline.v1",
                "title": "一次示范，取走最上层纸杯",
                "resolution": [WIDTH, HEIGHT],
                "fps": FPS,
                "duration_s": round(cursor, 3),
                "evidence_boundary": (
                    "Run08 tactile is raw-UVC/offline shadow; retry is scripted; "
                    "single-cup outcome is front-RGB/operator confirmed; robot movie ends at hold."
                ),
                "scenes": timeline,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    OUT_SRT.write_text("\n".join(srt_parts), encoding="utf-8")
    rows = [
        "# 纸杯任务 framework 演示片脚本",
        "",
        "配色：青色 = agentic，金色 = 物体相对位姿，粉色 = 触觉监工，绿色 = 通过，红色 = 重试。",
        "",
        "| 时间 | 画面 | 字幕 | 证据 |",
        "|---|---|---|---|",
    ]
    for item in timeline:
        rows.append(
            f"| {item['start_s']:.1f}–{item['end_s']:.1f}s | {item['scene']} | "
            f"{item['subtitle']} | {item['evidence']} |"
        )
    rows.extend(
        [
            "",
            "## 证据边界",
            "",
            "Run08 两路触觉是 raw UVC 加离线 absdiff 的 shadow 诊断；错误低探后的重试为脚本预设，",
            "没有把 shadow 结果写成实时触觉闭环。单杯成功来自前视 RGB 的 operator visual confirmation。",
            "机器人录像止于 LIFT / HOLD；TRANSPORT / PLACE / RELEASE 只在人类示范中出现。",
            "",
        ]
    )
    OUT_SCRIPT.write_text("\n".join(rows), encoding="utf-8")


def render_movie(scenes: list[Scene]) -> None:
    total_frames = sum(int(round(scene.duration_s * FPS)) for scene in scenes)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(OUT_MP4),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin was not created")
    frame_no = 0
    try:
        for scene_index, scene in enumerate(scenes):
            count = int(round(scene.duration_s * FPS))
            for local_frame in range(count):
                local_s = local_frame / FPS
                image = scene.render(local_s, scene_index)
                # Gentle scene fades without sacrificing the continuous science layout.
                fade = min(1.0, local_s / 0.20, (scene.duration_s - local_s) / 0.20)
                if fade < 1.0:
                    dark = Image.new("RGB", image.size, BLACK)
                    image = Image.blend(dark, image.convert("RGB"), max(0.0, fade))
                process.stdin.write(np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes())
                frame_no += 1
                if frame_no % 300 == 0:
                    print(f"rendered {frame_no}/{total_frames} frames", flush=True)
    finally:
        process.stdin.close()
    code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg exited with code {code}")


def main() -> int:
    for path in (HUMAN_STAGE, PERCEPTION, RIM_RELATION, ROBOT_RUN):
        if not path.exists():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenes = build_scenes()
    write_metadata(scenes)
    render_movie(scenes)
    print(OUT_MP4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
