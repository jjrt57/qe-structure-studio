"""Interactive 3D structure viewer: builds a self-contained HTML page rendered by 3Dmol.js."""

from __future__ import annotations

import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from pymatgen.core import Element, Structure
from scipy.spatial import cKDTree

THREEDMOL_VERSION = "2.5.5"
THREEDMOL_URLS = (
    f"https://cdn.jsdelivr.net/npm/3dmol@{THREEDMOL_VERSION}/build/3Dmol-min.js",
    f"https://unpkg.com/3dmol@{THREEDMOL_VERSION}/build/3Dmol-min.js",
)
MAX_VIEW_ATOMS = 6000
BOUNDARY_TOL = 1e-3  # fractional distance from a cell face that counts as "on the face"
BOND_SCALE = 1.15  # bonded if d < BOND_SCALE * (r_cov,i + r_cov,j)
STYLES = {"Ball and stick": "ball_stick", "Space-filling": "spacefill", "Sticks": "sticks"}
COLOR_SCHEMES = ("VESTA", "Jmol")
DEFAULT_ZOOM = 1.18

_AXIS_COLORS = ("#C8372D", "#23874B", "#2459C4")  # a, b, c
_EDGE_COLOR = "#1D3A55"
_SUPERCELL_EDGE_COLOR = "#7C93A8"
_FALLBACK_COLOR = "#8C9AA8"


class ViewerTooLarge(ValueError):
    """Raised when the requested view would contain too many atoms to render smoothly."""


@lru_cache(maxsize=1)
def _color_tables() -> dict:
    try:
        import pymatgen.core
        from monty.serialization import loadfn

        path = Path(pymatgen.core.__file__).resolve().parent.parent / "vis" / "ElementColorSchemes.yaml"
        return loadfn(path)
    except Exception:
        return {}


def element_colors(symbols, scheme: str = "VESTA") -> dict[str, str]:
    """Hex colors per element from pymatgen's VESTA or Jmol tables."""
    table = _color_tables().get(scheme, {})
    colors = {}
    for symbol in symbols:
        rgb = table.get(symbol)
        colors[symbol] = "#{:02X}{:02X}{:02X}".format(*(int(c) for c in rgb)) if rgb else _FALLBACK_COLOR
    return colors


def _radius(symbol: str) -> float:
    return float(covalent_radii[atomic_numbers.get(symbol, 0)]) or 1.0


def estimated_atom_count(structure: Structure, supercell=(1, 1, 1)) -> int:
    return len(structure) * int(np.prod(supercell))


def _display_atoms(structure: Structure, supercell, boundary: bool):
    """Atoms to draw: the (super)cell contents plus images of atoms on faces and corners."""
    reps = tuple(int(n) for n in supercell)
    cell = structure * reps if reps != (1, 1, 1) else structure
    frac = cell.frac_coords - np.floor(cell.frac_coords + 1e-8)
    frac[np.abs(frac) < 1e-8] = 0.0
    symbols = [site.specie.symbol for site in cell]
    labels = cell.site_properties.get("qe_label") or symbols
    lattice = cell.lattice.matrix
    out_symbols, out_labels, out_frac, out_cart = [], [], [], []
    for index, f in enumerate(frac):
        options = []
        for axis in range(3):
            shifts = [0.0]
            if boundary and f[axis] < BOUNDARY_TOL:
                shifts.append(1.0)
            elif boundary and f[axis] > 1.0 - BOUNDARY_TOL:
                shifts.append(-1.0)
            options.append(shifts)
        for shift in itertools.product(*options):
            shifted = f + np.array(shift)
            out_symbols.append(symbols[index])
            out_labels.append(labels[index])
            out_frac.append(shifted * np.array(reps))  # in units of the original cell
            out_cart.append(shifted @ lattice)
    return out_symbols, out_labels, np.array(out_frac), np.array(out_cart), lattice


def _bond_pairs(symbols: list[str], cart: np.ndarray) -> np.ndarray:
    if len(symbols) < 2:
        return np.empty((0, 2), dtype=int)
    radii = np.array([_radius(s) for s in symbols])
    metal = np.array([Element(s).is_metal for s in symbols])
    pairs = cKDTree(cart).query_pairs(r=BOND_SCALE * 2 * radii.max(), output_type="ndarray")
    if len(pairs) == 0:
        return pairs
    i, j = pairs[:, 0], pairs[:, 1]
    distance = np.linalg.norm(cart[i] - cart[j], axis=1)
    keep = (distance > 0.1) & (distance <= BOND_SCALE * (radii[i] + radii[j]))
    if not metal.all():  # in compounds, draw cation-anion bonds but not metal-metal ones
        keep &= ~(metal[i] & metal[j])
    return pairs[keep]


def _style(style: str, symbol: str, color: str) -> dict:
    radius = _radius(symbol)
    if style == "spacefill":
        return {"sphere": {"radius": round(max(0.35, 1.05 * radius), 3), "color": color}}
    if style == "sticks":
        return {"stick": {"radius": 0.15, "color": color}, "sphere": {"radius": 0.15, "color": color}}
    ball = min(0.75, max(0.2, 0.4 * radius))
    return {"sphere": {"radius": round(ball, 3), "color": color}, "stick": {"radius": 0.11, "color": color}}


def _box(lattice: np.ndarray, radius: float, axis_colors: bool, dashed: bool, color: str) -> list[dict]:
    edges = []
    for corner in itertools.product((0, 1), repeat=3):
        for axis in range(3):
            if corner[axis]:
                continue
            end = list(corner)
            end[axis] = 1
            start_xyz = np.array(corner) @ lattice
            end_xyz = np.array(end) @ lattice
            from_origin = corner == (0, 0, 0)
            edges.append({
                "s": np.round(start_xyz, 4).tolist(),
                "e": np.round(end_xyz, 4).tolist(),
                "r": round(radius * (1.6 if axis_colors and from_origin else 1.0), 4),
                "c": _AXIS_COLORS[axis] if axis_colors and from_origin else color,
                "d": dashed,
            })
    return edges


def _view_quaternion(lattice: np.ndarray) -> list[float]:
    """Camera rotation for a textbook view: c up, b to the right, a toward the viewer.

    Returns the (x, y, z, w) quaternion that rotates model coordinates into view
    coordinates, as used by 3Dmol's getView/setView.
    """
    a, _, c = np.asarray(lattice, dtype=float)
    e_c = c / np.linalg.norm(c)
    a_perp = a - (a @ e_c) * e_c
    if np.linalg.norm(a_perp) < 1e-8:
        a_perp = np.cross(e_c, [1.0, 0.0, 0.0] if abs(e_c[0]) < 0.9 else [0.0, 1.0, 0.0])
    e_a = a_perp / np.linalg.norm(a_perp)
    e_b = np.cross(e_c, e_a)
    toward_camera = e_a + 0.55 * e_b + 0.45 * e_c
    z = toward_camera / np.linalg.norm(toward_camera)
    y = e_c - (e_c @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    m = np.array([x, y, z])
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = [(m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s, 0.25 / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    return [round(float(v), 6) for v in q]


def _json_for_script(data) -> str:
    """JSON that is safe to embed inside a <script> element."""
    return (json.dumps(data, separators=(",", ":"))
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))


def build_viewer_html(
    structure: Structure,
    *,
    style: str = "ball_stick",
    supercell=(1, 1, 1),
    show_cell: bool = True,
    show_boundary: bool = True,
    show_labels: bool = False,
    spin: bool = False,
    color_scheme: str = "VESTA",
    filename: str = "structure",
    description: str = "Crystal structure",
) -> str:
    """Return a complete HTML document that renders ``structure`` with 3Dmol.js."""
    if estimated_atom_count(structure, supercell) > MAX_VIEW_ATOMS:
        raise ViewerTooLarge(
            f"A {supercell[0]}×{supercell[1]}×{supercell[2]} supercell of {len(structure)} atoms "
            f"is more than the viewer's {MAX_VIEW_ATOMS}-atom limit.")
    symbols, labels, frac, cart, super_lattice = _display_atoms(structure, supercell, show_boundary)
    bonds = _bond_pairs(symbols, cart)
    neighbors: list[list[int]] = [[] for _ in symbols]
    for i, j in bonds:
        neighbors[int(i)].append(int(j))
        neighbors[int(j)].append(int(i))

    colors = element_colors(sorted(set(symbols)), color_scheme)
    atoms = []
    for index, (symbol, label) in enumerate(zip(symbols, labels)):
        fx, fy, fz = frac[index]
        atoms.append({
            "e": symbol,
            "l": label,
            "p": np.round(cart[index], 4).tolist(),
            "b": neighbors[index],
            "i": f"{label} ({fx:.3f}, {fy:.3f}, {fz:.3f})",
        })

    unit_lattice = structure.lattice.matrix
    extent = float(np.linalg.norm(super_lattice, axis=1).max())
    edge_radius = max(0.02, 0.0075 * extent)
    edges, axis_labels = [], []
    if show_cell:
        edges += _box(unit_lattice, edge_radius, axis_colors=True, dashed=False, color=_EDGE_COLOR)
        if tuple(supercell) != (1, 1, 1):
            edges += _box(super_lattice, edge_radius * 0.8, axis_colors=False, dashed=True,
                          color=_SUPERCELL_EDGE_COLOR)
        centre = 0.5 * unit_lattice.sum(axis=0)
        for axis, name in enumerate("abc"):
            midpoint = 0.5 * unit_lattice[axis]
            outward = midpoint - centre
            outward /= np.linalg.norm(outward) or 1.0
            position = midpoint + outward * max(0.45, 0.06 * extent)
            axis_labels.append({"t": name, "p": np.round(position, 4).tolist(), "c": _AXIS_COLORS[axis]})

    data = {
        "atoms": atoms,
        "styles": {symbol: _style(style, symbol, colors[symbol]) for symbol in colors},
        "edges": edges,
        "axisLabels": axis_labels,
        "labels": bool(show_labels) and len(atoms) <= 400,
        "spin": bool(spin),
        "quat": _view_quaternion(unit_lattice),
        "zoom": DEFAULT_ZOOM,
        "filename": filename,
    }
    return (_PAGE
            .replace("__DATA__", _json_for_script(data))
            .replace("__PRIMARY_URL__", THREEDMOL_URLS[0])
            .replace("__FALLBACK_URL__", _json_for_script(THREEDMOL_URLS[1]))
            .replace("__ARIA__", json.dumps(description)[1:-1].replace("<", "").replace(">", "")))


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --paper: #F2F5F8; --ink: #12283D; --grid: rgba(18, 40, 61, .05); --grid-major: rgba(18, 40, 61, .085); }
  html, body { margin: 0; height: 100%; background: transparent; }
  body { font: 13px/1.4 Archivo, "Helvetica Neue", Arial, sans-serif; color: var(--ink); }
  .stage { position: relative; height: 100%; overflow: hidden; border-radius: 10px;
    background-color: var(--paper);
    background-image:
      linear-gradient(var(--grid-major) 1px, transparent 1px),
      linear-gradient(90deg, var(--grid-major) 1px, transparent 1px),
      linear-gradient(var(--grid) 1px, transparent 1px),
      linear-gradient(90deg, var(--grid) 1px, transparent 1px);
    background-size: 120px 120px, 120px 120px, 24px 24px, 24px 24px;
    background-position: -1px -1px; }
  #viewer { position: absolute; inset: 0; }
  .tools { position: absolute; top: 10px; right: 10px; display: flex; gap: 6px; z-index: 5; }
  .tools button { appearance: none; cursor: pointer; font: 500 12px/1 inherit; font-family: inherit;
    color: var(--ink); background: rgba(255, 255, 255, .88); border: 1px solid rgba(18, 40, 61, .22);
    border-radius: 6px; padding: 7px 10px; }
  .tools button:hover { background: #fff; border-color: rgba(18, 40, 61, .5); }
  .tools button:focus-visible { outline: 2px solid #2459C4; outline-offset: 2px; }
  .hint { position: absolute; left: 12px; bottom: 9px; z-index: 5; pointer-events: none;
    font-size: 11.5px; color: rgba(18, 40, 61, .6); }
  .status { position: absolute; inset: 0; z-index: 6; display: grid; place-items: center;
    padding: 24px; text-align: center; background: var(--paper); font-size: 14px; }
  .status[hidden] { display: none; }
</style>
</head>
<body>
<div class="stage">
  <div id="viewer" role="img" aria-label="__ARIA__"></div>
  <div class="tools">
    <button id="reset" type="button">Reset view</button>
    <button id="png" type="button">Save PNG</button>
  </div>
  <div class="hint">Drag to rotate, scroll to zoom, right-drag to pan. Hover an atom for its fractional coordinates.</div>
  <div class="status" id="status" hidden></div>
</div>
<script src="__PRIMARY_URL__"></script>
<script>
(function () {
  "use strict";
  var D = __DATA__;
  var statusBox = document.getElementById("status");
  function fail(message) { statusBox.textContent = message; statusBox.hidden = false; }
  function vec(p) { return { x: p[0], y: p[1], z: p[2] }; }

  function start() {
    if (!window.$3Dmol) {
      fail("The 3D viewer (3Dmol.js) could not be loaded. Check your connection or content blocker. Downloads still work.");
      return;
    }
    var viewer = $3Dmol.createViewer(document.getElementById("viewer"),
      { backgroundColor: "white", backgroundAlpha: 0, antialias: true, orthographic: true });
    viewer.setBackgroundColor(0xffffff, 0);

    var model = viewer.addModel();
    model.addAtoms(D.atoms.map(function (a, i) {
      return { elem: a.e, x: a.p[0], y: a.p[1], z: a.p[2], serial: i, index: i,
               bonds: a.b, bondOrder: a.b.map(function () { return 1; }), info: a.i };
    }));
    Object.keys(D.styles).forEach(function (el) { viewer.setStyle({ elem: el }, D.styles[el]); });

    D.edges.forEach(function (e) {
      viewer.addCylinder({ start: vec(e.s), end: vec(e.e), radius: e.r, color: e.c,
                           fromCap: 1, toCap: 1, dashed: e.d, dashLength: 0.35, gapLength: 0.25 });
    });
    D.axisLabels.forEach(function (l) {
      viewer.addLabel(l.t, { position: vec(l.p), font: "Archivo, Helvetica, Arial, sans-serif", fontSize: 19,
                             fontColor: l.c, backgroundColor: "#F2F5F8", backgroundOpacity: 0.75,
                             borderThickness: 0, inFront: true, alignment: "center" });
    });
    if (D.labels) {
      D.atoms.forEach(function (a) {
        viewer.addLabel(a.l, { position: vec(a.p), fontSize: 11, fontColor: "#12283D", backgroundColor: "#FFFFFF",
                               backgroundOpacity: 0.8, borderThickness: 0, inFront: true, alignment: "center" });
      });
    }

    var hoverLabel = null;
    viewer.setHoverable({}, true, function (atom, v) {
      if (hoverLabel) { v.removeLabel(hoverLabel); }
      hoverLabel = v.addLabel(atom.info, { position: { x: atom.x, y: atom.y, z: atom.z }, fontSize: 12,
        fontColor: "#FFFFFF", backgroundColor: "#12283D", backgroundOpacity: 0.92, borderThickness: 0,
        inFront: true, alignment: "bottomCenter" });
      v.render();
    }, function (atom, v) {
      if (hoverLabel) { v.removeLabel(hoverLabel); hoverLabel = null; v.render(); }
    });
    if (typeof viewer.setHoverDuration === "function") { viewer.setHoverDuration(60); }

    viewer.zoomTo();
    var view = viewer.getView();
    view[4] = D.quat[0]; view[5] = D.quat[1]; view[6] = D.quat[2]; view[7] = D.quat[3];
    viewer.setView(view);
    viewer.zoomTo();
    viewer.zoom(D.zoom);
    viewer.render();
    var home = viewer.getView();

    var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (D.spin && !reduceMotion) { viewer.spin("y", 0.45); }

    document.getElementById("reset").addEventListener("click", function () {
      viewer.setView(home);
      viewer.render();
    });
    document.getElementById("png").addEventListener("click", function () {
      viewer.setBackgroundColor(0xffffff, 1);
      viewer.render();
      var uri = viewer.pngURI();
      viewer.setBackgroundColor(0xffffff, 0);
      viewer.render();
      var link = document.createElement("a");
      link.href = uri;
      link.download = D.filename + ".png";
      document.body.appendChild(link);
      link.click();
      link.remove();
    });
    window.__viewerReady = true;
  }

  if (window.$3Dmol) {
    start();
  } else {
    var fallback = document.createElement("script");
    fallback.src = __FALLBACK_URL__;
    fallback.onload = start;
    fallback.onerror = start;
    document.head.appendChild(fallback);
  }
})();
</script>
</body>
</html>
"""
