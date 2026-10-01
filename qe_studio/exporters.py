"""Write structures to the file formats offered in the app and the CLI."""

from __future__ import annotations

import contextlib
import io
import math
import re
import warnings
import zipfile
from dataclasses import dataclass
from typing import Callable

from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifWriter

# Target k-point "length" (Å) used to suggest a Monkhorst-Pack grid in pw.x inputs.
K_POINT_LENGTH = 25.0


@dataclass(frozen=True)
class ExportFormat:
    key: str
    label: str
    suffix: str  # appended to the file stem, e.g. ".cif" or "_geometry.in"
    description: str
    mime: str = "text/plain"

    def filename(self, stem: str) -> str:
        return f"{stem}{self.suffix}"


FORMATS: list[ExportFormat] = [
    ExportFormat("cif", "CIF", ".cif",
                 "Crystallographic Information File listing every atom (space group P1). "
                 "Opens in VESTA, Mercury, OVITO and most crystal software."),
    ExportFormat("cif_sym", "CIF, symmetrized", "_sym.cif",
                 "Space group found by spglib at the chosen tolerance. Only symmetry-unique "
                 "atoms are listed, in the conventional cell with idealized positions."),
    ExportFormat("xyz", "XYZ", ".xyz",
                 "Cartesian coordinates in Å. Every molecular viewer reads it, but plain XYZ "
                 "stores no unit cell."),
    ExportFormat("extxyz", "Extended XYZ", ".extxyz",
                 "XYZ plus lattice vectors, periodic boundaries and the DFT energy (eV). "
                 "Reads straight into ASE and OVITO."),
    ExportFormat("poscar", "VASP POSCAR", ".vasp",
                 "Fractional coordinates grouped by element. Rename the file to POSCAR to "
                 "run VASP; VESTA opens .vasp files directly."),
    ExportFormat("qe", "Quantum ESPRESSO input", ".in",
                 "A pw.x scf input with ibrav = 0, the cell in Å and crystal coordinates. "
                 "Pseudopotential names, ecutwfc and k-points are placeholders to edit."),
    ExportFormat("xsf", "XCrySDen XSF", ".xsf",
                 "Periodic structure file for XCrySDen, also read by VESTA and OVITO."),
    ExportFormat("pdb", "PDB", ".pdb",
                 "Protein Data Bank format with a CRYST1 cell record. Coordinates are "
                 "rounded to 0.001 Å."),
    ExportFormat("lammps", "LAMMPS data", ".lmp",
                 "atom_style atomic in metal units, with masses. The cell is rotated into "
                 "LAMMPS' triclinic convention (a along x)."),
    ExportFormat("castep", "CASTEP cell", ".cell",
                 "LATTICE_CART and POSITIONS_FRAC blocks for CASTEP."),
    ExportFormat("aims", "FHI-aims geometry", "_geometry.in",
                 "lattice_vector and atom lines in Å. Rename to geometry.in for FHI-aims."),
    ExportFormat("json", "pymatgen JSON", ".json",
                 "A serialized pymatgen Structure, as used by the Materials Project.",
                 "application/json"),
]
FORMATS_BY_KEY: dict[str, ExportFormat] = {fmt.key: fmt for fmt in FORMATS}


@dataclass(frozen=True)
class ExportContext:
    title: str = ""
    energy_ev: float | None = None
    symprec: float = 0.01
    qe_template: dict | None = None  # parsed pw.x input whose settings the QE export keeps


def _symbols(structure: Structure) -> list[str]:
    return [site.specie.symbol for site in structure]


def _labels(structure: Structure) -> list[str]:
    """QE species labels when available (Fe1, Fe2, ...), element symbols otherwise."""
    labels = structure.site_properties.get("qe_label")
    return list(labels) if labels and all(labels) else _symbols(structure)


def _one_line(text: str) -> str:
    """A comment line that no reader will mistake for key=value data."""
    return re.sub(r"[\r\n=\"']", " ", text).strip() or "structure"


def _ase_atoms(structure: Structure, energy_ev: float | None = None):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms = Atoms(symbols=_symbols(structure), positions=structure.cart_coords,
                  cell=structure.lattice.matrix, pbc=True)
    if energy_ev is not None:
        atoms.calc = SinglePointCalculator(atoms, energy=float(energy_ev))
    return atoms


def _ase_write(atoms, fmt: str, **kwargs) -> str:
    from ase.io import write

    buffer = io.StringIO()
    # Some ASE writers (CASTEP) print setup chatter to stdout; keep it out of logs.
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        write(buffer, atoms, format=fmt, **kwargs)
    return buffer.getvalue()


def _cif(structure: Structure, ctx: ExportContext) -> str:
    return str(CifWriter(structure))


def _cif_symmetrized(structure: Structure, ctx: ExportContext) -> str:
    try:
        return str(CifWriter(structure, symprec=ctx.symprec))
    except Exception as exc:  # spglib can fail on badly distorted cells
        return (f"# Symmetry detection failed at symprec = {ctx.symprec} Å ({exc}).\n"
                "# Written in space group P1 instead.\n" + str(CifWriter(structure)))


def _xyz(structure: Structure, ctx: ExportContext) -> str:
    lines = [str(len(structure)), _one_line(ctx.title)]
    for symbol, (x, y, z) in zip(_symbols(structure), structure.cart_coords):
        lines.append(f"{symbol:<2s} {x:15.8f} {y:15.8f} {z:15.8f}")
    return "\n".join(lines) + "\n"


def _extxyz(structure: Structure, ctx: ExportContext) -> str:
    return _ase_write(_ase_atoms(structure, ctx.energy_ev), "extxyz")


def _poscar(structure: Structure, ctx: ExportContext) -> str:
    return structure.get_sorted_structure().to(fmt="poscar")


_TEMPLATE_DROPPED = {"ibrav", "nat", "ntyp", "a", "b", "c", "cosab", "cosac", "cosbc", "celldm",
                     "space_group", "uniqueb", "origin_choice", "rhombohedral"}
_TEMPLATE_REWRITTEN_CARDS = {"ATOMIC_SPECIES", "CELL_PARAMETERS", "ATOMIC_POSITIONS"}


def _template_fits(structure: Structure, template: dict | None) -> bool:
    if not template or not template.get("species"):
        return False
    return set(_labels(structure)) <= {species["label"] for species in template["species"]}


def _pwscf_from_template(structure: Structure, ctx: ExportContext) -> str:
    """The user's own pw.x input with the cell written out explicitly (ibrav = 0)."""
    template = ctx.qe_template
    labels = _labels(structure)
    species = template["species"]
    if_pos = template.get("if_pos") or []
    keep_flags = len(if_pos) == len(structure) and any(flags != [1, 1, 1] for flags in if_pos)
    lines = [f"! {_one_line(ctx.title)}",
             "! Written by QE Structure Studio from a pw.x input: the same settings, with the cell",
             "! written out explicitly (ibrav = 0) and positions in crystal coordinates."]
    for name, entries in template["namelists"].items():
        lines.append(f"&{name.upper()}")
        if name == "system":
            lines += ["  ibrav = 0", f"  nat = {len(structure)}", f"  ntyp = {len(species)}"]
        for key, raw in entries:
            if name == "system" and (key in _TEMPLATE_DROPPED or key.startswith("celldm(")):
                continue
            if name == "cell" and key == "cell_dofree" and raw.strip("'\" ").lower() == "ibrav":
                raw = "'all'"
                lines.insert(3, "! cell_dofree = 'ibrav' needs ibrav /= 0, so it was changed to 'all'.")
            lines.append(f"  {key} = {raw}")
        lines.append("/")
    lines += ["", "ATOMIC_SPECIES"]
    lines += [f"  {sp['label']:<4s} {sp['mass']:>10s}  {sp['pseudo']}" for sp in species]
    lines += ["", "CELL_PARAMETERS angstrom"]
    lines += [f"  {v[0]:15.9f} {v[1]:15.9f} {v[2]:15.9f}" for v in structure.lattice.matrix]
    lines += ["", "ATOMIC_POSITIONS crystal"]
    for index, (label, f) in enumerate(zip(labels, structure.frac_coords)):
        flags = "  {} {} {}".format(*if_pos[index]) if keep_flags else ""
        lines.append(f"  {label:<4s} {f[0]:14.10f} {f[1]:14.10f} {f[2]:14.10f}{flags}")
    for card in template["cards"]:
        if card["name"] not in _TEMPLATE_REWRITTEN_CARDS:
            lines += ["", card["header"]] + [f"  {line}" for line in card["lines"]]
    return "\n".join(lines) + "\n"


def _pwscf(structure: Structure, ctx: ExportContext) -> str:
    if _template_fits(structure, ctx.qe_template):
        return _pwscf_from_template(structure, ctx)
    symbols, labels = _symbols(structure), _labels(structure)
    species = list(dict.fromkeys(labels))
    element_of = dict(zip(labels, symbols))
    kpoints = [max(1, math.ceil(K_POINT_LENGTH * b / (2 * math.pi)))
               for b in structure.lattice.reciprocal_lattice.abc]
    prefix = re.sub(r"[^A-Za-z0-9]", "", structure.composition.reduced_formula) or "structure"
    lines = [
        f"! {_one_line(ctx.title)}",
        "! Written by QE Structure Studio. The pseudopotential file names, ecutwfc and",
        "! K_POINTS below are placeholders: set them for your system before running.",
        "&CONTROL",
        "  calculation = 'scf'",
        f"  prefix      = '{prefix}'",
        "  outdir      = './tmp'",
        "  pseudo_dir  = './pseudo'",
        "/",
        "&SYSTEM",
        "  ibrav   = 0",
        f"  nat     = {len(structure)}",
        f"  ntyp    = {len(species)}",
        "  ecutwfc = 40.0",
        "/",
        "&ELECTRONS",
        "  conv_thr = 1.0d-8",
        "/",
        "",
        "ATOMIC_SPECIES",
    ]
    for label in species:
        element = element_of[label]
        lines.append(f"  {label:<4s} {float(Element(element).atomic_mass):10.4f}  {element}.UPF")
    lines += ["", "CELL_PARAMETERS angstrom"]
    lines += [f"  {v[0]:15.9f} {v[1]:15.9f} {v[2]:15.9f}" for v in structure.lattice.matrix]
    lines += ["", "ATOMIC_POSITIONS crystal"]
    lines += [f"  {label:<4s} {f[0]:14.10f} {f[1]:14.10f} {f[2]:14.10f}"
              for label, f in zip(labels, structure.frac_coords)]
    lines += ["", "K_POINTS automatic", "  {} {} {} 0 0 0".format(*kpoints), ""]
    return "\n".join(lines)


def _xsf(structure: Structure, ctx: ExportContext) -> str:
    return structure.to(fmt="xsf")


def _pdb(structure: Structure, ctx: ExportContext) -> str:
    return _ase_write(_ase_atoms(structure), "proteindatabank")


def _lammps(structure: Structure, ctx: ExportContext) -> str:
    order = list(dict.fromkeys(_symbols(structure)))
    return _ase_write(_ase_atoms(structure), "lammps-data", specorder=order,
                      atom_style="atomic", masses=True, units="metal")


def _castep(structure: Structure, ctx: ExportContext) -> str:
    return _ase_write(_ase_atoms(structure), "castep-cell")


def _aims(structure: Structure, ctx: ExportContext) -> str:
    lines = [f"# {_one_line(ctx.title)}"]
    lines += [f"lattice_vector {v[0]:15.9f} {v[1]:15.9f} {v[2]:15.9f}" for v in structure.lattice.matrix]
    lines += [f"atom {x:15.9f} {y:15.9f} {z:15.9f} {symbol}"
              for symbol, (x, y, z) in zip(_symbols(structure), structure.cart_coords)]
    return "\n".join(lines) + "\n"


def _json(structure: Structure, ctx: ExportContext) -> str:
    return structure.to(fmt="json")


_WRITERS: dict[str, Callable[[Structure, ExportContext], str]] = {
    "cif": _cif, "cif_sym": _cif_symmetrized, "xyz": _xyz, "extxyz": _extxyz,
    "poscar": _poscar, "qe": _pwscf, "xsf": _xsf, "pdb": _pdb, "lammps": _lammps,
    "castep": _castep, "aims": _aims, "json": _json,
}


def export_structure(structure: Structure, key: str, *, title: str = "",
                     energy_ev: float | None = None, symprec: float = 0.01,
                     qe_template: dict | None = None) -> str:
    """Return the text of ``structure`` in the format ``key`` (see ``FORMATS``).

    ``qe_template`` is the ``input_data`` of a parsed pw.x input; the Quantum ESPRESSO
    export then keeps that input's settings instead of writing placeholders.
    """
    if key not in _WRITERS:
        raise KeyError(f"Unknown format '{key}'. Choose from: {', '.join(_WRITERS)}")
    return _WRITERS[key](structure, ExportContext(title=title, energy_ev=energy_ev, symprec=symprec,
                                                  qe_template=qe_template))


def export_bundle(structure: Structure, stem: str, *, title: str = "",
                  energy_ev: float | None = None, symprec: float = 0.01,
                  qe_template: dict | None = None) -> bytes:
    """A zip archive with the structure in every format."""
    buffer = io.BytesIO()
    errors = []
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for fmt in FORMATS:
            try:
                text = export_structure(structure, fmt.key, title=title, energy_ev=energy_ev,
                                        symprec=symprec, qe_template=qe_template)
            except Exception as exc:
                errors.append(f"{fmt.label}: {exc}")
                continue
            archive.writestr(fmt.filename(stem), text)
        if errors:
            archive.writestr("EXPORT_ERRORS.txt", "\n".join(errors) + "\n")
    return buffer.getvalue()


def export_trajectory(structures: list[Structure], energies_ev: list[float | None],
                      infos: list[dict] | None = None) -> str:
    """Multi-frame extended XYZ with one frame per ionic step."""
    from ase.io import write

    images = []
    for index, (structure, energy) in enumerate(zip(structures, energies_ev)):
        atoms = _ase_atoms(structure, energy)
        info = infos[index] if infos else {"step": index}
        atoms.info.update({k: v for k, v in info.items() if v is not None})
        images.append(atoms)
    buffer = io.StringIO()
    write(buffer, images, format="extxyz")
    return buffer.getvalue()


def format_for_filename(filename: str, symmetrize: bool = False) -> ExportFormat:
    """Guess the export format from an output file name (used by the CLI)."""
    name = filename.lower()
    base = name.rsplit("/", 1)[-1]
    if base.endswith(".cif"):
        return FORMATS_BY_KEY["cif_sym" if symmetrize else "cif"]
    if base in ("poscar", "contcar") or base.endswith((".vasp", ".poscar")):
        return FORMATS_BY_KEY["poscar"]
    if base.endswith("geometry.in"):
        return FORMATS_BY_KEY["aims"]
    for suffix, key in ((".extxyz", "extxyz"), (".xyz", "xyz"), (".in", "qe"), (".pwi", "qe"),
                        (".xsf", "xsf"), (".pdb", "pdb"), (".lmp", "lammps"), (".data", "lammps"),
                        (".cell", "castep"), (".json", "json")):
        if base.endswith(suffix):
            return FORMATS_BY_KEY[key]
    raise ValueError(f"Can't tell the format from '{filename}'. Use an extension such as "
                     ".cif, .xyz, .extxyz, .vasp, .in, .xsf, .pdb, .lmp, .cell or .json.")
