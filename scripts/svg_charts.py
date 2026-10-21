"""Small static SVG line charts for the WS3 analysis report (no plotting library needed).

Static figures for a Markdown report: legend on top, direct labels at line ends, thin lines,
recessive grid, text in ink colours (never the series colour). Two series at most per panel.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from html import escape

# Reference palette, light mode (validated categorical slots 1 and 2).
SERIES_COLOURS = ("#2a78d6", "#eb6834")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
TARGET = "#52514e"

PANEL_W, PANEL_H = 380, 230
PAD_L, PAD_R, PAD_T, PAD_B = 52, 124, 34, 42
LEGEND_H = 34


@dataclass(frozen=True)
class Series:
    name: str
    ys: Sequence[float]
    lo: Sequence[float] | None = None  # optional confidence band
    hi: Sequence[float] | None = None


@dataclass(frozen=True)
class Panel:
    title: str
    xs: Sequence[float]
    series: Sequence[Series]
    x_label: str
    y_min: float
    y_max: float
    percent: bool = True
    y_step: float | None = None  # gridline spacing; a quarter of the range by default
    target: tuple[float, str] | None = None  # (y, label) reference line
    x_format: Callable[[float], str] = field(default=lambda x: f"{x:g}")


def _fmt(v: float, percent: bool) -> str:
    if percent:
        return f"{v:.0%}" if abs(v * 100 - round(v * 100)) < 1e-9 else f"{v:.1%}"
    return f"{v:g}" if abs(v * 10 - round(v * 10)) < 1e-9 else f"{v:.2f}"


def _panel(p: Panel, ox: float, oy: float) -> list[str]:
    w, h = PANEL_W - PAD_L - PAD_R, PANEL_H - PAD_T - PAD_B
    x0, x1 = min(p.xs), max(p.xs)

    def sx(x: float) -> float:
        return ox + PAD_L + (x - x0) / ((x1 - x0) or 1) * w

    def sy(y: float) -> float:
        y = min(max(y, p.y_min), p.y_max)
        return oy + PAD_T + h - (y - p.y_min) / (p.y_max - p.y_min) * h

    out = [
        f'<text x="{ox + PAD_L}" y="{oy + 18}" font-size="13" font-weight="600" '
        f'fill="{INK}">{escape(p.title)}</text>'
    ]
    step = p.y_step or (p.y_max - p.y_min) / 4
    for i in range(round((p.y_max - p.y_min) / step) + 1):
        y = p.y_min + step * i
        out.append(
            f'<line x1="{ox + PAD_L}" x2="{ox + PAD_L + w}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{ox + PAD_L - 6}" y="{sy(y) + 4:.1f}" font-size="11" text-anchor="end" '
            f'fill="{INK_SECONDARY}">{_fmt(y, p.percent)}</text>'
        )
    for x in p.xs:
        out.append(
            f'<text x="{sx(x):.1f}" y="{oy + PAD_T + h + 16}" font-size="11" '
            f'text-anchor="middle" fill="{INK_SECONDARY}">{escape(p.x_format(x))}</text>'
        )
    out.append(
        f'<text x="{ox + PAD_L + w / 2}" y="{oy + PANEL_H - 6}" font-size="11" '
        f'text-anchor="middle" fill="{INK_SECONDARY}">{escape(p.x_label)}</text>'
    )
    if p.target is not None:
        ty, label = p.target
        out.append(
            f'<line x1="{ox + PAD_L}" x2="{ox + PAD_L + w}" y1="{sy(ty):.1f}" y2="{sy(ty):.1f}" '
            f'stroke="{TARGET}" stroke-width="1" stroke-dasharray="4 3"/>'
        )
        out.append(
            f'<text x="{ox + PAD_L + 4}" y="{sy(ty) - 4:.1f}" font-size="10" '
            f'fill="{INK_SECONDARY}">{escape(label)}</text>'
        )
    ends: list[tuple[float, str]] = []
    for s, colour in zip(p.series, SERIES_COLOURS, strict=False):
        if s.lo is not None and s.hi is not None:
            top = [f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(p.xs, s.hi, strict=True)]
            bottom = [f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(p.xs, s.lo, strict=True)]
            out.append(
                f'<polygon points="{" ".join(top + bottom[::-1])}" fill="{colour}" '
                f'fill-opacity="0.15" stroke="none"/>'
            )
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(p.xs, s.ys, strict=True))
        out.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y in zip(p.xs, s.ys, strict=True):
            out.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{colour}" '
                f'stroke="{SURFACE}" stroke-width="2"/>'
            )
        ends.append((sy(s.ys[-1]), f"{s.name} {_fmt(s.ys[-1], p.percent)}"))
    # Direct labels at the line ends, nudged apart when they would overlap.
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 13:
            ends[i] = (ends[i - 1][0] + 13, ends[i][1])
    for y, label in ends:
        out.append(
            f'<text x="{ox + PAD_L + w + 18}" y="{y + 4:.1f}" font-size="10.5" '
            f'fill="{INK}">{escape(label)}</text>'
        )
    return out


def chart(title: str, panels: Sequence[Panel], columns: int = 2) -> str:
    """One SVG with a shared legend and the panels as small multiples."""
    rows = -(-len(panels) // columns)
    width = PANEL_W * min(columns, len(panels))
    height = 30 + LEGEND_H + PANEL_H * rows
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" '
        f'role="img" aria-label="{escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        f'<text x="12" y="22" font-size="15" font-weight="700" fill="{INK}">{escape(title)}</text>',
    ]
    # A single series needs no legend: the title names it.
    names = [s.name for s in panels[0].series] if len(panels[0].series) > 1 else []
    x = 12.0
    for name, colour in zip(names, SERIES_COLOURS, strict=False):
        out.append(
            f'<line x1="{x}" x2="{x + 18}" y1="44" y2="44" stroke="{colour}" stroke-width="2"/>'
        )
        out.append(f'<circle cx="{x + 9}" cy="44" r="4" fill="{colour}"/>')
        out.append(f'<text x="{x + 24}" y="48" font-size="12" fill="{INK}">{escape(name)}</text>')
        x += 32 + 7 * len(name)
    for i, panel in enumerate(panels):
        ox, oy = PANEL_W * (i % columns), 30 + LEGEND_H + PANEL_H * (i // columns)
        out += _panel(panel, ox, oy)
    out.append("</svg>")
    return "\n".join(out) + "\n"
