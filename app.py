"""QE Structure Studio.

Read a Quantum ESPRESSO pw.x output file, inspect the structure in 3D and
download it as CIF, XYZ, POSCAR and other formats.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from qe_studio.analysis import CELL_SETTINGS, formula_html, space_group_html, standardized, symmetry_summary
from qe_studio.exporters import FORMATS, FORMATS_BY_KEY, export_bundle, export_structure, export_trajectory
from qe_studio.parser import BOHR_TO_ANGSTROM, QEParseError, QEResult, decode_output, parse_qe_file
from qe_studio.viewer import (COLOR_SCHEMES, MAX_VIEW_ATOMS, STYLES, ViewerTooLarge, build_viewer_html,
                              element_colors, estimated_atom_count)

APP_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = APP_DIR / "examples"
EXAMPLES = {
    "Rutile TiO₂": "tio2_rutile_vc-relax.out",
    "Silicon": "si_vc-relax.out",
    "Water molecule": "h2o_relax.out",
    "ZnO input file": "zno_wurtzite.in",
}
DEFAULT_EXAMPLE = "Rutile TiO₂"
GITHUB_URL = ""  # e.g. "https://github.com/<you>/qe-structure-studio"; linked in the footer when set
VIEWER_HEIGHT = 540
PREVIEW_LINES = 150

RUN_DESCRIPTIONS = {
    "vc-relax": "Variable-cell relaxation",
    "relax": "Relaxation at fixed cell",
    "scf": "Single-point SCF calculation",
    "nscf/bands": "Non-self-consistent calculation",
    "md": "Molecular dynamics",
    "vc-md": "Variable-cell molecular dynamics",
}
INPUT_DESCRIPTIONS = {
    "scf": "an SCF calculation", "nscf": "a non-self-consistent calculation",
    "bands": "a band-structure calculation", "relax": "a relaxation at fixed cell",
    "vc-relax": "a variable-cell relaxation", "md": "a molecular-dynamics run",
    "vc-md": "a variable-cell molecular-dynamics run",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:ital,wdth,wght@0,62..125,100..900;1,62..125,100..900&display=swap');
:root { --qs-line: #2C4D6A; --qs-muted: #A0B6C9; --qs-accent: #FFC940; --qs-ink: #11263A; }
[data-testid="stMainBlockContainer"], .block-container { max-width: 1480px; padding-top: 4.6rem; padding-bottom: 3rem; }
[data-testid="stDecoration"] { display: none; }
.qs-mast { display: flex; gap: 1rem; align-items: flex-start; margin: 0 0 .35rem; }
.qs-mark { flex: none; margin-top: .2rem; }
.qs-title { font-family: Archivo, sans-serif; font-stretch: 125%; font-weight: 650; letter-spacing: -.012em;
  font-size: clamp(1.65rem, 2.5vw, 2.3rem); line-height: 1.05; }
.qs-lede { margin: .5rem 0 0; color: var(--qs-muted); max-width: 64ch; font-size: 1rem; line-height: 1.5; }
.qs-file { font-stretch: 112%; font-weight: 600; font-size: 1.3rem; line-height: 1.25; overflow-wrap: anywhere; }
.qs-summary { color: var(--qs-muted); margin: .25rem 0 0; max-width: 80ch; }
.qs-facts { display: flex; flex-wrap: wrap; margin: .2rem 0 .6rem;
  border-top: 1px solid var(--qs-line); border-bottom: 1px solid var(--qs-line); }
.qs-facts > div { flex: 1 1 auto; padding: .75rem 1.1rem .8rem 1.1rem; border-left: 1px solid var(--qs-line); min-width: 8.5rem; }
.qs-facts > div:first-child { border-left: 0; padding-left: 0; }
.qs-facts dt { font-size: .8rem; color: var(--qs-muted); margin: 0 0 .3rem; }
.qs-facts dd { margin: 0; font-size: 1.1rem; font-weight: 560; font-variant-numeric: tabular-nums; white-space: nowrap; }
@media (max-width: 640px) { .qs-facts > div { border-left: 0; padding-left: 0; min-width: 45%; } .qs-facts dd { white-space: normal; } }
.qs-facts dd small { display: block; margin-top: .2rem; font-size: .8rem; font-weight: 400; color: var(--qs-muted); }
.qs-h3 { font-stretch: 112%; font-weight: 600; font-size: 1.08rem; }
.qs-legend { display: flex; flex-wrap: wrap; gap: .35rem 1.1rem; margin: .6rem 0 0; font-size: .9rem; color: var(--qs-muted); }
.qs-legend span { display: inline-flex; align-items: center; gap: .45rem; }
.qs-legend i { display: inline-block; width: .85rem; height: .85rem; border-radius: 50%;
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, .28); }
.qs-sg { font-size: 1.6rem; font-stretch: 105%; line-height: 1.2; }
.ovl { text-decoration: overline; text-decoration-thickness: .06em; }
.qs-empty { border: 1px dashed var(--qs-line); border-radius: .5rem; padding: 1.4rem 1.5rem; margin-top: 1rem; max-width: 70ch; }
.qs-empty p { margin: 0 0 .4rem; }
.qs-muted { color: var(--qs-muted); }
.qs-foot { color: var(--qs-muted); font-size: .86rem; border-top: 1px solid var(--qs-line); margin-top: 2.2rem; padding-top: 1rem; }
[data-testid="stBaseButton-primary"], button[kind="primary"] { color: var(--qs-ink) !important; font-weight: 600; }
[data-testid="stBaseButton-primary"] p, button[kind="primary"] p { color: var(--qs-ink) !important; }
</style>
"""

MARK = """<svg class="qs-mark" width="40" height="40" viewBox="0 0 40 40" aria-hidden="true">
<g fill="none" stroke="#FFC940" stroke-width="1.5" stroke-linejoin="round">
<path d="M14 5h20v20H14z" stroke-opacity=".5"/><path d="M5 14l9-9M25 14l9-9M25 34l9-9M5 34l9-9" stroke-opacity=".5"/>
<path d="M5 14h20v20H5z"/></g>
<g fill="#FFC940"><circle cx="5" cy="14" r="2.4"/><circle cx="25" cy="14" r="2.4"/><circle cx="25" cy="34" r="2.4"/>
<circle cx="5" cy="34" r="2.4"/></g><circle cx="19.5" cy="19.5" r="3.4" fill="#E4EDF5"/></svg>"""


# ----------------------------------------------------------------------------- cached work

@st.cache_data(show_spinner="Reading the pw.x output…", max_entries=12)
def load_result(data: bytes) -> QEResult:
    return parse_qe_file(decode_output(data))


@st.cache_data(show_spinner=False, max_entries=64)
def get_structure(key: str, step: int, setting: str, symprec: float, wrap: bool, _result: QEResult):
    return standardized(_result.frames[step].structure(wrap=wrap), setting, symprec)


@st.cache_data(show_spinner=False, max_entries=64)
def get_symmetry(key: str, symprec: float, _structure):
    return symmetry_summary(_structure, symprec)


@st.cache_data(show_spinner=False, max_entries=128)
def get_export(key: str, fmt: str, title: str, energy_ev, symprec: float, _structure, _qe_template=None) -> str:
    return export_structure(_structure, fmt, title=title, energy_ev=energy_ev, symprec=symprec,
                            qe_template=_qe_template)


@st.cache_data(show_spinner=False, max_entries=16)
def get_bundle(key: str, stem: str, title: str, energy_ev, symprec: float, _structure, _qe_template=None) -> bytes:
    return export_bundle(_structure, stem, title=title, energy_ev=energy_ev, symprec=symprec,
                         qe_template=_qe_template)


@st.cache_data(show_spinner=False, max_entries=8)
def get_trajectory(digest: str, wrap: bool, _result: QEResult) -> str:
    frames = _result.frames
    infos = [{"step": i, "pressure_kbar": f.pressure_kbar, "total_force_ry_bohr": f.total_force}
             for i, f in enumerate(frames)]
    return export_trajectory([f.structure(wrap=wrap) for f in frames], [f.energy_ev for f in frames], infos)


@st.cache_data(show_spinner=False, max_entries=32)
def get_viewer(key: str, style: str, supercell: tuple, show_cell: bool, show_boundary: bool,
               show_labels: bool, spin: bool, scheme: str, filename: str, description: str, _structure) -> str:
    return build_viewer_html(_structure, style=style, supercell=supercell, show_cell=show_cell,
                             show_boundary=show_boundary, show_labels=show_labels, spin=spin,
                             color_scheme=scheme, filename=filename, description=description)


# ----------------------------------------------------------------------------- helpers

def render_html(document: str, height: int) -> None:
    if hasattr(st, "iframe"):
        st.iframe(document, height=height)
    else:  # older Streamlit
        import streamlit.components.v1 as components
        components.html(document, height=height)


def clean_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("._")[:80]


def make_stem(source_name: str, step: int, n_frames: int, setting: str) -> str:
    name = Path(source_name).name
    name = re.sub(r"\.gz$", "", name, flags=re.I)
    name = re.sub(r"\.(out|log|txt|pwo|stdout|in|pwi|inp)$", "", name, flags=re.I)
    stem = clean_stem(name) or "structure"
    if n_frames > 1:
        stem += "_final" if step == n_frames - 1 else f"_step{step}"
    if setting != "as_computed":
        stem += "_conv" if setting == "conventional" else "_prim"
    return stem


def minus(text: str) -> str:
    """Typographic minus signs for displayed numbers."""
    return text.replace("-", "\u2212")


def describe_run(result: QEResult, is_example: bool) -> str:
    if result.source == "input":
        data = result.input_data or {}
        run = INPUT_DESCRIPTIONS.get(result.calculation, f"calculation = '{html.escape(result.calculation)}'")
        if data.get("ibrav") == 0:
            cell = f"ibrav = 0, with the cell from CELL_PARAMETERS in {result.cell_units}"
        else:
            cell = (f"ibrav = {data.get('ibrav')} ({data.get('ibrav_name', '')}), with the cell from "
                    f"{data.get('lattice_source', 'celldm')}")
        text = f"pw.x input file for {run}. {cell}."
        return text + (" This is one of the bundled examples." if is_example else "")
    steps = len(result.frames) - 1
    plural = "s" if steps != 1 else ""
    text = RUN_DESCRIPTIONS.get(result.calculation, result.calculation)
    if result.is_relaxation:
        if result.converged:
            text += f", converged after {steps} ionic step{plural}"
        elif result.converged is False:
            text += f", stopped after {steps} ionic step{plural} without converging"
        else:
            text += f", {steps} ionic step{plural} read"
    elif result.calculation in ("md", "vc-md"):
        text += f", {steps} step{plural}"
    text += "."
    if result.qe_version:
        text += f" Quantum ESPRESSO {html.escape(result.qe_version)}."
    if is_example:
        text += " This is one of the bundled examples."
    return text


def frame_caption(result: QEResult, step: int) -> str:
    last = len(result.frames) - 1
    frame = result.frames[step]
    if step == 0:
        text = "Step 0 is the input geometry."
    elif step == last:
        text = f"Step {step} is the final structure." if result.converged else f"Step {step} is the last geometry in the file."
    else:
        text = f"Step {step} of {last}."
    if frame.energy_ry is None:
        text += " pw.x did not compute an energy for it."
    return text


def facts_html(structure, sym: dict | None, frame, symprec: float, result: QEResult | None = None) -> str:
    reduced, factor = structure.composition.get_reduced_formula_and_factor()
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    if sym:
        group = space_group_html(sym["symbol"])
        group_note = f"No. {sym['number']}, {sym['crystal_system']}"
    else:
        group, group_note = "Not found", f"at {symprec} Å tolerance"
    if frame.energy_ry is not None:
        energy = minus(f"{frame.energy_ry:.6f} Ry")
        energy_note = minus(f"{frame.energy_ev:.4f} eV, {frame.nat} atoms")
    else:
        energy, energy_note = "Not computed", "pw.x stopped before this SCF"
    items = [
        ("Formula", formula_html(reduced), f"Z = {factor:g}"),
        ("Space group", group, group_note),
        ("a, b, c (Å)", f"{a:.4f}, {b:.4f}, {c:.4f}", None),
        ("α, β, γ (°)", f"{alpha:.2f}, {beta:.2f}, {gamma:.2f}", None),
        ("Volume (Å³)", f"{structure.volume:.3f}", f"{structure.density:.4f} g/cm³, {len(structure)} atoms"),
        ("Total energy", energy, energy_note),
    ]
    if result is not None and result.source == "input":
        data = result.input_data or {}
        cutoff, extras = data.get("ecutwfc"), []
        if isinstance(data.get("ecutrho"), (int, float)):
            extras.append(f"ecutrho {data['ecutrho']:g} Ry")
        extras.append("k-points " + html.escape(str(data.get("kpoints", "not set")).split(",")[0].replace(" grid", "")))
        items[-1] = ("Cutoff (ecutwfc)", f"{cutoff:g} Ry" if isinstance(cutoff, (int, float)) else "Not set",
                     ", ".join(extras))
    cells = "".join(
        f"<div><dt>{label}</dt><dd>{value}{f'<small>{note}</small>' if note else ''}</dd></div>"
        for label, value, note in items)
    return f'<dl class="qs-facts">{cells}</dl>'


def render_input_settings(result: QEResult) -> None:
    data = result.input_data or {}
    summary = [f"ibrav = {data.get('ibrav')}: {data.get('ibrav_name', '')}."]
    if data.get("ibrav") != 0:
        values = [f"celldm({i + 1}) = {v:.6g}" for i, v in enumerate(data.get("celldm", [])) if v]
        summary.append(", ".join(values) + ".")
    if result.alat_bohr:
        summary.append(f"Lattice parameter {result.alat_bohr:.6f} bohr = {result.alat_bohr * BOHR_TO_ANGSTROM:.6f} Å.")
    summary.append(f"Positions in {result.position_units} units. K-points: {data.get('kpoints', 'not set')}.")
    st.text(" ".join(summary))
    left, right = st.columns(2, gap="large")
    with left:
        for name, entries in data.get("namelists", {}).items():
            st.caption(f"&{name.upper()}" + ("" if entries else " (empty)"))
            if entries:
                st.dataframe(pd.DataFrame(entries, columns=["Variable", "Value"]), hide_index=True, width="stretch")
    with right:
        if data.get("species"):
            st.caption("ATOMIC_SPECIES")
            st.dataframe(pd.DataFrame([{"Label": sp["label"], "Element": sp["element"], "Mass": sp["mass"],
                                        "Pseudopotential": sp["pseudo"]} for sp in data["species"]]),
                         hide_index=True, width="stretch")
        for card in data.get("cards", []):
            if card["name"] not in ("ATOMIC_SPECIES", "ATOMIC_POSITIONS", "CELL_PARAMETERS"):
                st.caption(card["header"])
                st.code("\n".join(card["lines"]) or "(no lines)", language=None)


def input_details(result: QEResult) -> dict:
    data = result.input_data or {}
    return {
        "File type": "pw.x input",
        "calculation": result.calculation,
        "ibrav": f"{data.get('ibrav')} ({data.get('ibrav_name', '')})",
        "Cell from": str(data.get("lattice_source", "")),
        "Lattice parameter alat (bohr)": f"{result.alat_bohr:.6f}" if result.alat_bohr else "not set",
        "CELL_PARAMETERS units": result.cell_units or "not used",
        "ATOMIC_POSITIONS units": str(result.position_units),
        "nat, ntyp": f"{data.get('nat')}, {data.get('ntyp')}",
        "K_POINTS": str(data.get("kpoints", "not set")),
    }


def legend_html(structure, scheme: str) -> str:
    counts = structure.composition.get_el_amt_dict()
    colors = element_colors(list(counts), scheme)
    spans = "".join(f'<span><i style="background:{colors[el]}"></i>{el} &times; {counts[el]:g}</span>'
                    for el in counts)
    return f'<div class="qs-legend" aria-label="Element colors">{spans}</div>'


def step_chart(df: pd.DataFrame, column: str, title: str, step: int, color: str = "#8CC4FF"):
    data = df[["Step", column]].dropna()
    base = alt.Chart(data).encode(
        x=alt.X("Step:Q", title="Ionic step", axis=alt.Axis(tickMinStep=1, format="d")),
        y=alt.Y(f"{column}:Q", title=title, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("Step:Q"), alt.Tooltip(f"{column}:Q", title=title, format=".4f")],
    )
    line = base.mark_line(color=color, strokeWidth=2) + base.mark_point(color=color, filled=True, size=45)
    marker = alt.Chart(pd.DataFrame({"Step": [step]})).mark_rule(color="#FFC940", strokeDash=[4, 3]).encode(x="Step:Q")
    return (line + marker).properties(height=230)


# ----------------------------------------------------------------------------- page

st.set_page_config(
    page_title="QE Structure Studio",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "Read Quantum ESPRESSO pw.x output, inspect the structure in 3D and export it."},
)
st.html(CSS)

st.markdown(
    f"""<div class="qs-mast">{MARK}<div>
<div class="qs-title" role="heading" aria-level="1">QE Structure Studio</div>
<p class="qs-lede">Open a Quantum ESPRESSO pw.x input or output file, check the structure in 3D, and
download it as CIF, XYZ, POSCAR or nine other formats.</p></div></div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Settings")
    symprec = st.select_slider(
        "Symmetry tolerance (Å)", options=[0.001, 0.003, 0.01, 0.03, 0.1], value=0.01,
        help="Distance tolerance for spglib. It sets the space group shown, the symmetrized CIF "
             "and the conventional and primitive cells.")
    wrap = st.toggle(
        "Wrap atoms into the unit cell", value=True,
        help="Map fractional coordinates into [0, 1). Turn off to keep molecules that cross "
             "the cell boundary in one piece.")
    st.divider()
    st.markdown("### Command line")
    st.caption("The same converter runs without the web app, on inputs and outputs:")
    st.code("python qe2cif.py -i relax.out -o relaxed.cif", language="bash")
    st.caption("The output extension picks the format: .cif, .xyz, .extxyz, .vasp, .in, .xsf, .pdb, .lmp, .cell, .json")


def _on_upload() -> None:
    if st.session_state.get("upload") is not None:
        st.session_state["example"] = None


if "example" not in st.session_state:
    st.session_state["example"] = DEFAULT_EXAMPLE

col_upload, col_examples = st.columns([1.55, 1], gap="large", vertical_alignment="bottom")
with col_upload:
    uploaded = st.file_uploader(
        "pw.x input or output file", key="upload", on_change=_on_upload,
        help="An input file (any ibrav with celldm or A, B, C, or CELL_PARAMETERS), or the output of an "
             "scf, relax, vc-relax or md run. Plain text or .gz.")
with col_examples:
    example = st.pills("Or open an example", list(EXAMPLES), key="example")

if example:
    source_name, is_example = EXAMPLES[example], True
    data = (EXAMPLE_DIR / source_name).read_bytes()
elif uploaded is not None:
    source_name, is_example = uploaded.name, False
    data = uploaded.getvalue()
else:
    st.markdown(
        """<div class="qs-empty"><p>Upload a pw.x input or output file, or open one of the examples.</p>
<p class="qs-muted">Inputs can use any ibrav with celldm or A, B, C, or CELL_PARAMETERS. Outputs from scf,
relax, vc-relax and md runs all work, whichever units pw.x printed.</p></div>""",
        unsafe_allow_html=True,
    )
    st.stop()

digest = hashlib.sha1(data).hexdigest()[:16]
try:
    result = load_result(data)
except QEParseError as exc:
    st.error(f"{source_name} couldn't be read. {exc}", icon=":material/error:")
    st.stop()
except Exception as exc:  # unexpected: show it rather than a blank page
    st.error(f"{source_name} couldn't be read because of an unexpected error: {exc}", icon=":material/error:")
    st.stop()

n_frames = len(result.frames)
is_input = result.source == "input"
warning_count = sum(1 for level, _ in result.notes if level == "warning")

st.space("small")
st.markdown(
    f'<div class="qs-file" role="heading" aria-level="2">{html.escape(source_name)}</div>'
    f'<p class="qs-summary">{describe_run(result, is_example)}</p>',
    unsafe_allow_html=True,
)
with st.container(horizontal=True, gap="small"):
    if is_input:
        st.badge("Input file", icon=":material/description:", color="blue")
    elif result.converged is True:
        st.badge("Converged", icon=":material/check_circle:", color="green")
    elif result.converged is False:
        st.badge("Not converged", icon=":material/warning:", color="orange")
    if not is_input and not result.job_done:
        st.badge("Run unfinished", icon=":material/error:", color="red")
    if warning_count:
        st.badge(f"{warning_count} parser note{'s' if warning_count > 1 else ''}", icon=":material/info:", color="orange")

col_step, col_cell = st.columns([1.6, 1], gap="large", vertical_alignment="top")
with col_step:
    if n_frames > 1:
        step = st.slider("Ionic step", min_value=0, max_value=n_frames - 1, value=n_frames - 1,
                         key=f"step-{digest}",
                         help="Step 0 is the input geometry. The last step is the final structure.")
        st.caption(frame_caption(result, step))
    else:
        step = 0
        st.caption("The structure defined by this input file." if is_input else "This file contains a single geometry.")
with col_cell:
    setting_label = st.segmented_control(
        "Cell", list(CELL_SETTINGS), default="As computed", key="setting", required=True,
        help="Conventional and primitive cells are standardized by spglib, which also idealizes "
             "positions within the tolerance. The choice applies to the 3D view and every download.")
setting = CELL_SETTINGS[setting_label or "As computed"]

frame = result.frames[step]
key = f"{digest}:{step}:{setting}:{symprec}:{wrap}"
structure, setting_message = get_structure(key, step, setting, symprec, wrap, result)
if setting_message:
    st.info(setting_message, icon=":material/info:")
sym = get_symmetry(key, symprec, structure)
reduced_formula = structure.composition.reduced_formula

st.markdown(facts_html(structure, sym, frame, symprec, result), unsafe_allow_html=True)

col_view, col_export = st.columns([2.15, 1], gap="large")

with col_view:
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.markdown('<div class="qs-h3" role="heading" aria-level="3">Structure</div>', unsafe_allow_html=True)
        view_options = st.popover("View options", icon=":material/tune:")
    with view_options:
        style_label = st.segmented_control("Style", list(STYLES), default="Ball and stick",
                                           key="style", required=True)
        st.caption("Repeat the cell along a, b and c")
        rep_a, rep_b, rep_c = st.columns(3)
        n_a = rep_a.number_input("a", min_value=1, max_value=6, value=1, key="rep_a")
        n_b = rep_b.number_input("b", min_value=1, max_value=6, value=1, key="rep_b")
        n_c = rep_c.number_input("c", min_value=1, max_value=6, value=1, key="rep_c")
        show_cell = st.toggle("Unit cell and axes", value=True, key="show_cell")
        show_boundary = st.toggle("Repeat atoms on cell faces and corners", value=True, key="show_boundary")
        show_labels = st.toggle("Atom labels", value=False, key="show_labels")
        spin = st.toggle("Rotate slowly", value=False, key="spin")
        scheme = st.segmented_control("Colors", list(COLOR_SCHEMES), default="VESTA", key="scheme",
                                      required=True) or "VESTA"
    supercell = (int(n_a), int(n_b), int(n_c))
    if estimated_atom_count(structure, supercell) > MAX_VIEW_ATOMS and supercell != (1, 1, 1):
        st.warning(f"That supercell would draw more than {MAX_VIEW_ATOMS:,} atoms, so the single cell is shown.",
                   icon=":material/warning:")
        supercell = (1, 1, 1)
    try:
        viewer_page = get_viewer(key, STYLES[style_label or "Ball and stick"], supercell, show_cell, show_boundary,
                                 show_labels, spin, scheme, make_stem(source_name, step, n_frames, setting),
                                 f"3D model of {reduced_formula}", structure)
        render_html(viewer_page, VIEWER_HEIGHT)
        st.markdown(legend_html(structure, scheme), unsafe_allow_html=True)
    except ViewerTooLarge:
        st.info(f"This cell has {len(structure):,} atoms, more than the 3D view draws smoothly "
                f"({MAX_VIEW_ATOMS:,}). Every download still works.", icon=":material/info:")

with col_export:
    with st.container(border=True):
        st.markdown('<div class="qs-h3" role="heading" aria-level="3">Download</div>', unsafe_allow_html=True)
        fmt_key = st.selectbox(
            "Format", [f.key for f in FORMATS], key="fmt",
            format_func=lambda k: f"{FORMATS_BY_KEY[k].label} "
                                  f"({FORMATS_BY_KEY[k].suffix.replace('_sym', '').replace('_geometry', 'geometry')})")
        fmt = FORMATS_BY_KEY[fmt_key]
        qe_template = result.input_data if is_input and setting == "as_computed" else None
        if fmt_key == "qe" and qe_template:
            st.caption("Your input with the same settings, pseudopotentials and k-points, the cell written "
                       "out explicitly (ibrav = 0) and positions in crystal coordinates.")
        else:
            st.caption(fmt.description)
        default_stem = make_stem(source_name, step, n_frames, setting)
        stem = clean_stem(st.text_input("File name", value=default_stem, key=f"stem-{default_stem}",
                                        help="Without the extension; it's added for the chosen format.")) or default_stem
        filename = f"{stem}_ibrav0.in" if fmt_key == "qe" and is_input else fmt.filename(stem)
        energy_ev = frame.energy_ev if setting == "as_computed" else None
        which = "input structure" if is_input else ("final structure" if step == n_frames - 1 else f"ionic step {step}")
        cell_note = "" if setting == "as_computed" else f", {setting} cell"
        title = f"{reduced_formula} {which}{cell_note} from {source_name}"
        try:
            text = get_export(f"{key}:{fmt_key}", fmt_key, title, energy_ev, symprec, structure, qe_template)
        except Exception as exc:
            text = None
            st.error(f"Couldn't write {fmt.label}: {exc}", icon=":material/error:")
        if text is not None:
            lines = text.splitlines()
            preview = "\n".join(lines[:PREVIEW_LINES])
            if len(lines) > PREVIEW_LINES:
                preview += f"\n… {len(lines) - PREVIEW_LINES:,} more lines in the file"
            st.code(preview, language=None, height=290)
            st.download_button(f"Download {filename}", data=text, file_name=filename, mime=fmt.mime,
                               type="primary", icon=":material/download:", width="stretch",
                               on_click="ignore", key="download_main")
        st.download_button(
            "Download every format (.zip)",
            data=lambda: get_bundle(key, stem, title, energy_ev, symprec, structure, qe_template),
            file_name=f"{stem}_all_formats.zip", mime="application/zip",
            icon=":material/folder_zip:", width="stretch", on_click="ignore", key="download_zip")

tab_names = ["Lattice & symmetry", "Atoms"]
if n_frames > 1:
    tab_names.append("Relaxation" if result.is_relaxation else "Trajectory")
if is_input:
    tab_names.append("Input settings")
tab_names.append("Checks" if is_input else "Parser notes")
tabs = st.tabs(tab_names)

with tabs[0]:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        st.dataframe(
            pd.DataFrame({
                "Parameter": ["a (Å)", "b (Å)", "c (Å)", "α (°)", "β (°)", "γ (°)", "Volume (Å³)",
                              "Density (g/cm³)", "Atoms in cell"],
                "Value": [f"{a:.6f}", f"{b:.6f}", f"{c:.6f}", f"{alpha:.4f}", f"{beta:.4f}", f"{gamma:.4f}",
                          f"{structure.volume:.4f}", f"{structure.density:.4f}", str(len(structure))],
            }),
            hide_index=True, width="stretch")
        st.caption("Lattice vectors (Å)")
        st.dataframe(
            pd.DataFrame(structure.lattice.matrix, columns=["x", "y", "z"], index=["a", "b", "c"]),
            width="stretch", column_config={col: st.column_config.NumberColumn(format="%.6f") for col in "xyz"})
    with right:
        if sym:
            st.markdown(
                f'<div class="qs-sg">{space_group_html(sym["symbol"])}</div>'
                f'<p class="qs-muted">Space group {sym["number"]}, {sym["crystal_system"]} '
                f'({sym["lattice_type"]} lattice), point group {html.escape(sym["point_group"])}, '
                f'{sym["n_operations"]} symmetry operations in this cell. Hall symbol '
                f'{html.escape(sym["hall"])}. Tolerance {symprec} Å.</p>',
                unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(sym["sites"]), hide_index=True, width="stretch")
            st.caption("Wyckoff multiplicities refer to the conventional cell.")
        else:
            st.info("spglib found no space group at this tolerance. Try a larger symmetry tolerance "
                    "in the sidebar.", icon=":material/info:")

with tabs[1]:
    labels = structure.site_properties.get("qe_label") or [site.specie.symbol for site in structure]
    frac, cart = structure.frac_coords, structure.cart_coords
    atoms_df = pd.DataFrame({
        "Label": labels, "Element": [site.specie.symbol for site in structure],
        "Frac. a": frac[:, 0], "Frac. b": frac[:, 1], "Frac. c": frac[:, 2],
        "x (Å)": cart[:, 0], "y (Å)": cart[:, 1], "z (Å)": cart[:, 2],
    }, index=range(1, len(structure) + 1))
    numeric_columns = list(atoms_df.columns[2:])
    flags = (result.input_data or {}).get("if_pos") or [] if is_input and setting == "as_computed" else []
    if len(flags) == len(atoms_df) and any(f != [1, 1, 1] for f in flags):
        atoms_df["if_pos"] = [" ".join(str(v) for v in f) for f in flags]
    st.dataframe(atoms_df, width="stretch", height=min(460, 40 + 35 * len(atoms_df)),
                 column_config={col: st.column_config.NumberColumn(format="%.6f") for col in numeric_columns})
    if setting != "as_computed":
        st.caption("Standardized cells list element symbols; QE species labels such as Fe1 are "
                   "kept only in the cell as computed.")

if n_frames > 1:
    with tabs[2]:
        rows = []
        for index, f in enumerate(result.frames):
            fa, fb, fc = f.abc
            rows.append({"Step": index, "Energy (Ry)": f.energy_ry, "Energy (eV)": f.energy_ev,
                         "Pressure (kbar)": f.pressure_kbar, "Total force (Ry/bohr)": f.total_force,
                         "Volume (Å³)": f.volume, "a (Å)": fa, "b (Å)": fb, "c (Å)": fc})
        steps_df = pd.DataFrame(rows)
        known = steps_df["Energy (eV)"].dropna()
        charts = []
        if not known.empty:
            reference = known.iloc[-1]
            steps_df["dE"] = (steps_df["Energy (eV)"] - reference) * 1000 / result.final.nat
            charts.append(("dE", "ΔE (meV/atom)"))
        if steps_df["Total force (Ry/bohr)"].notna().any():
            charts.append(("Total force (Ry/bohr)", "Force (Ry/bohr)"))
        if steps_df["Pressure (kbar)"].notna().any():
            charts.append(("Pressure (kbar)", "Pressure (kbar)"))
        if steps_df["Volume (Å³)"].max() - steps_df["Volume (Å³)"].min() > 1e-6:
            charts.append(("Volume (Å³)", "Volume (Å³)"))
        for start in range(0, len(charts), 2):
            chart_cols = st.columns(2, gap="large")
            for col, (column, title) in zip(chart_cols, charts[start:start + 2]):
                with col:
                    st.altair_chart(step_chart(steps_df, column, title, step), width="stretch")
        st.caption("ΔE is measured from the last step with an energy. The dashed line marks the step shown above.")
        st.dataframe(steps_df.drop(columns=["dE"], errors="ignore"), hide_index=True, width="stretch",
                     column_config={col: st.column_config.NumberColumn(format="%.6f")
                                    for col in steps_df.columns if col not in ("Step", "dE")})
        st.download_button(
            "Download all steps (.extxyz)",
            data=lambda: get_trajectory(digest, wrap, result),
            file_name=f"{make_stem(source_name, 0, 1, 'as_computed')}_trajectory.extxyz",
            mime="text/plain", icon=":material/timeline:", on_click="ignore", key="download_traj")
        st.caption("One frame per ionic step with the lattice, energy (eV), pressure and total force. "
                   "Opens in OVITO and ASE's GUI as an animation.")

if is_input:
    with tabs[2]:
        render_input_settings(result)

with tabs[-1]:
    for level, message in result.notes:
        if level == "warning":
            st.warning(message, icon=":material/warning:")
        else:
            st.info(message, icon=":material/info:")
    if not result.notes:
        st.success(f"No problems found in this {'input' if is_input else 'output'}.", icon=":material/check_circle:")
    details = input_details(result) if is_input else {
        "Run type (inferred)": result.calculation,
        "Quantum ESPRESSO version": result.qe_version or "not found",
        "Geometries read": str(n_frames),
        "CELL_PARAMETERS units in file": result.cell_units or "none printed (cell from the header)",
        "ATOMIC_POSITIONS units in file": result.position_units or "none printed (positions from the header)",
        "alat (bohr)": f"{result.alat_bohr:.6f}" if result.alat_bohr else "not found",
        "Final SCF energy (Ry)": f"{result.final_scf_energy_ry:.8f}" if result.final_scf_energy_ry else "not run",
        "Finished with JOB DONE": "yes" if result.job_done else "no",
    }
    st.dataframe(pd.DataFrame({"Item": list(details), "Value": list(details.values())}),
                 hide_index=True, width="stretch")

repo = f' Source code on <a href="{html.escape(GITHUB_URL)}">GitHub</a>.' if GITHUB_URL else ""
st.markdown(
    '<div class="qs-foot">Symmetry by spglib, file formats by pymatgen and ASE, 3D view by 3Dmol.js. '
    "Unit conversions use the CODATA 2018 bohr radius." + repo + "</div>",
    unsafe_allow_html=True,
)
