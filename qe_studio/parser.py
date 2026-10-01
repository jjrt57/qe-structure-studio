"""Read crystal structures from Quantum ESPRESSO ``pw.x`` output files.

pw.x prints the geometry in several places and in several unit systems:

* the run header: ``crystal axes`` (units of alat) and the ``tau`` table of
  atomic positions (alat units). This is the only geometry in scf/nscf runs
  and the only cell in fixed-cell ``relax`` runs;
* one ``CELL_PARAMETERS`` / ``ATOMIC_POSITIONS`` pair per BFGS, MD or
  vc-relax step. CELL_PARAMETERS can be in ``alat``, ``bohr`` or
  ``angstrom``; ATOMIC_POSITIONS in ``alat``, ``bohr``, ``angstrom`` or
  ``crystal``;
* a ``Begin final coordinates`` block when a relaxation converges, followed
  (for vc-relax) by a final SCF with recomputed G-vectors.

Everything is converted to Å and fractional coordinates, and every ionic step
is kept, so the final structure and the whole relaxation can be inspected and
exported.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from pymatgen.core import Element, Lattice, Structure

BOHR_TO_ANGSTROM = 0.529177210903  # CODATA 2018
RY_TO_EV = 13.605693122994

_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eEdD][-+]?\d+)?"
_FLOAT_RE = re.compile(_FLOAT)
_RE_VERSION = re.compile(r"Program PWSCF\s+v\.?\s*(\S+)")
_RE_CELLDM1 = re.compile(r"celldm\(1\)\s*=\s*(" + _FLOAT + ")")
_RE_ALAT = re.compile(r"lattice parameter \(alat\)\s*=\s*(" + _FLOAT + ")")
_RE_NAT = re.compile(r"number of atoms/cell\s*=\s*(\d+)")
_RE_ENERGY = re.compile(r"^\s*!+\s*total energy\s*=\s*(" + _FLOAT + r")\s*Ry")
_RE_PRESSURE = re.compile(r"P\s*=\s*(" + _FLOAT + ")")
_RE_FORCE = re.compile(r"Total force\s*=\s*(" + _FLOAT + ")")
_RE_CARD_OPTION = re.compile(r"[({]\s*([A-Za-z_]+)\s*(?:=\s*(" + _FLOAT + r"))?\s*[)}]")
_RE_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{0,9}$")

# Geometries closer than this (Å) are treated as the same frame.
_SAME_GEOMETRY_TOL = 1e-4


class QEParseError(ValueError):
    """Raised when a file cannot be read as pw.x output."""


def _floats(text: str) -> list[float]:
    """Extract floats, including numbers that touch in fixed-width Fortran fields."""
    return [float(tok.replace("d", "e").replace("D", "e")) for tok in _FLOAT_RE.findall(text)]


@lru_cache(maxsize=None)
def element_from_label(label: str) -> str:
    """Map a QE species label ("Fe", "Fe1", "fe_up", "O2") to an element symbol.

    QE labels are the element symbol, optionally followed by a digit, a letter
    or ``_``/``-`` and a suffix, and are case-insensitive. The two-letter symbol
    is tried before the one-letter symbol ("Co1" -> Co, "C1" -> C).
    """
    if not _RE_LABEL.match(label):
        raise QEParseError(f"'{label[:20]}' is not a valid Quantum ESPRESSO species label.")
    letters = re.match(r"[A-Za-z]+", label).group(0)
    for size in (2, 1):
        if len(letters) >= size:
            symbol = letters[:size].capitalize()
            if Element.is_valid_symbol(symbol):
                return symbol
    raise QEParseError(
        f"Can't tell which element the species label '{label}' stands for. "
        "Labels should start with an element symbol, like Fe, Fe1 or Fe_up."
    )


@dataclass
class Frame:
    """One geometry printed by pw.x (the input structure or an ionic step)."""

    lattice: np.ndarray  # (3, 3) Å, rows are the lattice vectors a, b, c
    labels: list[str]  # QE species labels exactly as printed ("Fe1", "O", ...)
    frac_coords: np.ndarray  # (N, 3) fractional coordinates, not wrapped
    kind: str = "step"  # "input", "step" or "final"
    energy_ry: float | None = None
    pressure_kbar: float | None = None
    total_force: float | None = None  # Ry/bohr, as printed by pw.x
    is_final: bool = False
    species_elements: dict | None = None  # label -> element, from ATOMIC_SPECIES in input files

    @property
    def nat(self) -> int:
        return len(self.labels)

    @property
    def elements(self) -> list[str]:
        mapping = self.species_elements or {}
        return [mapping.get(label) or element_from_label(label) for label in self.labels]

    @property
    def energy_ev(self) -> float | None:
        return None if self.energy_ry is None else self.energy_ry * RY_TO_EV

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    @property
    def abc(self) -> tuple[float, float, float]:
        a, b, c = np.linalg.norm(self.lattice, axis=1)
        return float(a), float(b), float(c)

    def structure(self, wrap: bool = True) -> Structure:
        """Return a pymatgen Structure; QE labels are kept as the ``qe_label`` site property."""
        frac = np.array(self.frac_coords, dtype=float, copy=True)
        if wrap:
            frac -= np.floor(frac + 1e-8)
            frac[np.abs(frac) < 1e-8] = 0.0
        return Structure(
            Lattice(self.lattice),
            self.elements,
            frac,
            site_properties={"qe_label": list(self.labels)},
        )

    def same_geometry(self, other: "Frame", tol: float = _SAME_GEOMETRY_TOL) -> bool:
        if self.labels != other.labels or not np.allclose(self.lattice, other.lattice, atol=tol):
            return False
        delta = self.frac_coords - other.frac_coords
        delta -= np.round(delta)
        return bool(np.max(np.abs(delta @ self.lattice), initial=0.0) < tol)


@dataclass
class QEResult:
    """Everything read from one pw.x output file."""

    frames: list[Frame]
    calculation: str  # inferred: scf, nscf/bands, relax, vc-relax, md, vc-md
    converged: bool | None  # None when it can't be decided (e.g. md, truncated file)
    job_done: bool
    qe_version: str | None = None
    alat_bohr: float | None = None
    cell_units: str | None = None
    position_units: str | None = None
    final_scf_energy_ry: float | None = None
    final_scf_pressure_kbar: float | None = None
    notes: list[tuple[str, str]] = field(default_factory=list)  # (level, message)
    source: str = "output"  # "output" (pw.x output) or "input" (pw.x input file)
    input_data: dict | None = None  # namelists, cards and species of an input file

    @property
    def final(self) -> Frame:
        """The last geometry in the file (the relaxed structure for a converged relaxation)."""
        return self.frames[-1]

    @property
    def initial(self) -> Frame:
        return self.frames[0]

    @property
    def is_relaxation(self) -> bool:
        return self.calculation in ("relax", "vc-relax")

    def best_energy_ry(self) -> float | None:
        """Energy of the final structure: the final SCF if pw.x ran one, else the last step's."""
        if self.final_scf_energy_ry is not None:
            return self.final_scf_energy_ry
        return self.final.energy_ry


def _card_option(line: str, keyword: str) -> tuple[str | None, float | None]:
    """Units (and optional alat value) of a card header like ``CELL_PARAMETERS (alat= 9.9)``."""
    match = _RE_CARD_OPTION.search(line)
    if match:
        value = float(match.group(2).replace("d", "e").replace("D", "e")) if match.group(2) else None
        return match.group(1).lower(), value
    rest = line.strip()[len(keyword):].split()
    return (rest[0].lower() if rest else None), None


def _read_vectors(lines: list[str], start: int, count: int, after: str | None = None) -> list[list[float]] | None:
    """Read ``count`` lines of exactly three numbers (optionally only the part after ``after``)."""
    rows = []
    for line in lines[start:start + count]:
        text = line.split(after, 1)[1] if after and after in line else line
        values = _floats(text)
        if len(values) != 3:
            return None
        rows.append(values)
    return rows if len(rows) == count else None


def _read_positions(lines: list[str], start: int, nat: int | None) -> tuple[list[str], list[list[float]], int]:
    """Read an ATOMIC_POSITIONS block. Returns labels, coordinates and lines consumed."""
    labels, coords = [], []
    index = start
    while index < len(lines) and (nat is None or len(labels) < nat):
        parts = lines[index].split(None, 1)
        if len(parts) < 2 or not parts[0][0].isalpha():
            break
        values = _floats(parts[1])
        if len(values) < 3:
            break
        labels.append(parts[0])
        coords.append(values[:3])
        index += 1
    return labels, coords, index - start


def _looks_like_input_file(text: str) -> bool:
    lowered = text.lower()
    return ("&system" in lowered or "&control" in lowered) and "program pwscf" not in lowered


def parse_pw_output(text: str) -> QEResult:
    """Parse the text of a pw.x output file.

    Raises :class:`QEParseError` if no geometry can be found.
    """
    if _looks_like_input_file(text):
        raise QEParseError(
            "This is a pw.x input file, not an output; read it with parse_qe_file() "
            "or parse_pw_input()."
        )
    lines = text.splitlines()
    n_lines = len(lines)

    frames: list[Frame] = []
    notes: list[tuple[str, str]] = []
    qe_version = None
    alat_bohr = None  # bohr
    nat = None
    header_axes = None  # (3, 3) in units of alat, from the latest header
    cell = None  # (3, 3) Å, the most recent cell
    cell_units = position_units = None
    saw_cell_card = False
    in_final_block = after_final_block = False
    final_scf_energy = final_scf_pressure = None
    flags = dict.fromkeys(
        ["bfgs", "bfgs_converged", "max_steps", "bfgs_failed", "md", "bands",
         "scf_ok", "scf_failed", "job_done"], False)

    def add_frame(frame: Frame, dedupe: bool) -> None:
        if dedupe and frames and frames[-1].same_geometry(frame):
            frames[-1].is_final = frames[-1].is_final or frame.is_final
            return
        frames.append(frame)

    i = 0
    while i < n_lines:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("CELL_PARAMETERS"):
            units, alat_in_card = _card_option(stripped, "CELL_PARAMETERS")
            rows = _read_vectors(lines, i + 1, 3)
            if rows is None:
                notes.append(("warning", f"Skipped an incomplete CELL_PARAMETERS block (line {i + 1})."))
                i += 1
                continue
            if alat_in_card:
                alat_bohr = alat_in_card  # more digits than the header value
            vectors = np.array(rows)
            if units in (None, "alat"):
                if alat_bohr is None:
                    raise QEParseError("CELL_PARAMETERS is in units of alat, but alat was not found.")
                vectors = vectors * alat_bohr * BOHR_TO_ANGSTROM
            elif units == "bohr":
                vectors = vectors * BOHR_TO_ANGSTROM
            elif units != "angstrom":
                raise QEParseError(f"Unknown CELL_PARAMETERS units '{units}' (line {i + 1}).")
            cell, cell_units, saw_cell_card = vectors, units or "alat", True
            i += 4
            continue

        if stripped.startswith("ATOMIC_POSITIONS"):
            units, _ = _card_option(stripped, "ATOMIC_POSITIONS")
            labels, coords, used = _read_positions(lines, i + 1, nat)
            complete = labels and (nat is None or len(labels) == nat)
            if not complete:
                notes.append(("warning", f"Skipped an incomplete ATOMIC_POSITIONS block (line {i + 1})."))
                i += 1 + used
                continue
            if cell is None:
                raise QEParseError("Found ATOMIC_POSITIONS before any unit cell information; "
                                   "the file may be truncated or not a pw.x output.")
            coords = np.array(coords)
            if units == "crystal":
                frac = coords
            else:
                if units in (None, "alat"):
                    if alat_bohr is None:
                        raise QEParseError("ATOMIC_POSITIONS is in units of alat, but alat was not found.")
                    cart = coords * alat_bohr * BOHR_TO_ANGSTROM
                elif units == "bohr":
                    cart = coords * BOHR_TO_ANGSTROM
                elif units == "angstrom":
                    cart = coords
                else:
                    raise QEParseError(f"Unsupported ATOMIC_POSITIONS units '{units}' (line {i + 1}).")
                frac = cart @ np.linalg.inv(cell)
            for label in labels:
                element_from_label(label)  # fail early with a clear message
            position_units = units or "alat"
            frame = Frame(cell.copy(), labels, frac, kind="final" if in_final_block else "step",
                          is_final=in_final_block)
            add_frame(frame, dedupe=in_final_block)
            i += 1 + used
            continue

        if stripped.startswith("crystal axes:"):
            rows = _read_vectors(lines, i + 1, 3, after="=")
            if rows is not None:
                header_axes = np.array(rows)
            i += 4
            continue

        if stripped.startswith("site n.") and "positions (alat units)" in stripped:
            labels, taus = [], []
            j = i + 1
            while j < n_lines and "tau(" in lines[j] and (nat is None or len(labels) < nat):
                head, _, tail = lines[j].partition("tau(")
                values = _floats(tail.split("=", 1)[-1])
                if not head.split() or len(values) != 3:
                    break
                labels.append(head.split()[-1])
                taus.append(values)
                j += 1
            if labels and header_axes is not None and alat_bohr is not None and (nat is None or len(labels) == nat):
                scale = alat_bohr * BOHR_TO_ANGSTROM
                lattice = header_axes * scale
                frac = (np.array(taus) * scale) @ np.linalg.inv(lattice)
                for label in labels:
                    element_from_label(label)
                frame = Frame(lattice, labels, frac, kind="input" if not frames else "step")
                if not frames:
                    frames.append(frame)
                    cell = lattice
                elif not frames[-1].same_geometry(frame):
                    # A new header with a different geometry (e.g. several runs in one file).
                    frames.append(frame)
                    cell = lattice
            i = j
            continue

        # Scalar quantities and run status.
        match = _RE_ENERGY.match(line)
        if match:
            energy = float(match.group(1))
            if frames:
                if frames[-1].energy_ry is None:
                    frames[-1].energy_ry = energy
                elif after_final_block or frames[-1].is_final:
                    final_scf_energy = energy
        elif "total   stress" in line and "P=" in line:
            match = _RE_PRESSURE.search(line)
            if match and frames:
                pressure = float(match.group(1))
                if frames[-1].pressure_kbar is None:
                    frames[-1].pressure_kbar = pressure
                elif after_final_block or frames[-1].is_final:
                    final_scf_pressure = pressure
        elif "Total force =" in line:
            match = _RE_FORCE.search(line)
            if match and frames and frames[-1].total_force is None:
                frames[-1].total_force = float(match.group(1))
        elif "Begin final coordinates" in line:
            in_final_block = True
        elif "End final coordinates" in line:
            in_final_block, after_final_block = False, True
        elif "Program PWSCF" in line:
            match = _RE_VERSION.search(line)
            qe_version = match.group(1) if match else qe_version
        elif "celldm(1)=" in line:
            match = _RE_CELLDM1.search(line)
            if match and float(match.group(1)) > 0:
                alat_bohr = float(match.group(1))
        elif "lattice parameter (alat)" in line:
            match = _RE_ALAT.search(line)
            if match and alat_bohr is None:
                alat_bohr = float(match.group(1))
        elif "number of atoms/cell" in line:
            match = _RE_NAT.search(line)
            if match:
                nat = int(match.group(1))
        elif "BFGS Geometry Optimization" in line:
            flags["bfgs"] = True
        elif "bfgs converged" in line:
            flags["bfgs_converged"] = True
        elif "maximum number of steps has been reached" in line:
            flags["max_steps"] = True
        elif "bfgs failed" in line or "history already reset" in line:
            flags["bfgs_failed"] = True
        elif "Molecular Dynamics Calculation" in line or "Entering Dynamics" in line:
            flags["md"] = True
        elif "Band Structure Calculation" in line:
            flags["bands"] = True
        elif "convergence has been achieved" in line:
            flags["scf_ok"] = True
        elif "convergence NOT achieved" in line:
            flags["scf_failed"] = True
        elif "JOB DONE" in line:
            flags["job_done"] = True
        i += 1

    if not frames:
        if "Program PWSCF" not in text:
            raise QEParseError(
                "This doesn't look like pw.x output: there is no 'Program PWSCF' header. "
                "Use the standard output written by pw.x."
            )
        raise QEParseError(
            "No complete geometry was found. The run may have stopped before pw.x "
            "printed the structure."
        )

    # What kind of run was this?
    if flags["md"]:
        calculation = "vc-md" if saw_cell_card else "md"
    elif flags["bfgs"] or flags["bfgs_converged"] or flags["max_steps"]:
        calculation = "vc-relax" if saw_cell_card else "relax"
    elif flags["bands"]:
        calculation = "nscf/bands"
    else:
        calculation = "scf"

    if calculation in ("relax", "vc-relax"):
        if flags["bfgs_converged"]:
            converged = True
        elif flags["max_steps"] or flags["bfgs_failed"]:
            converged = False
        else:
            converged = None
    elif calculation in ("scf", "nscf/bands"):
        converged = False if flags["scf_failed"] else (True if flags["scf_ok"] else None)
    else:
        converged = None

    if flags["bfgs_converged"] and not frames[-1].is_final:
        frames[-1].is_final = True

    # Notes for the user.
    if not flags["job_done"]:
        notes.append(("warning", "pw.x did not finish: there is no 'JOB DONE.' line. "
                                 "The last complete geometry in the file is used."))
    relaxation = calculation in ("relax", "vc-relax")
    if relaxation and flags["max_steps"]:
        notes.append(("warning", "The relaxation reached nstep before converging. The last "
                                 "frame is the next geometry BFGS proposed; pw.x did not "
                                 "compute its energy. Restart from it with a larger nstep."))
    if relaxation and flags["bfgs_failed"]:
        notes.append(("warning", "BFGS stopped without converging (history reset or failure). "
                                 "Check forces and stress before trusting the final geometry."))
    if flags["scf_failed"]:
        notes.append(("warning", "At least one SCF cycle reported 'convergence NOT achieved'."))
    last_energy = frames[-1].energy_ry
    if final_scf_energy is not None and last_energy is not None:
        diff_mev = (final_scf_energy - last_energy) * RY_TO_EV * 1000 / frames[-1].nat
        if abs(diff_mev) > 1.0:
            notes.append(("warning", f"The final SCF at the relaxed cell differs from the last "
                                     f"BFGS step by {diff_mev:+.1f} meV/atom. A difference this "
                                     "large usually means the basis is not converged; raise "
                                     "ecutwfc/ecutrho to reduce Pulay stress."))
        else:
            notes.append(("info", f"The final SCF at the relaxed cell agrees with the last BFGS "
                                  f"step to {abs(diff_mev):.2f} meV/atom."))

    return QEResult(
        frames=frames,
        calculation=calculation,
        converged=converged,
        job_done=flags["job_done"],
        qe_version=qe_version,
        alat_bohr=alat_bohr,
        cell_units=cell_units,
        position_units=position_units,
        final_scf_energy_ry=final_scf_energy,
        final_scf_pressure_kbar=final_scf_pressure,
        notes=notes,
    )


def decode_output(data: bytes) -> str:
    """Decode uploaded bytes; gzip-compressed outputs are accepted too."""
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except OSError as exc:
            raise QEParseError("The file looks gzip-compressed but could not be decompressed.") from exc
    if b"\x00" in data[:4096]:
        raise QEParseError("This is a binary file. Use the text output written by pw.x.")
    return data.decode("utf-8", errors="replace")


def parse_qe_file(text: str) -> QEResult:
    """Parse a pw.x input or output file, whichever it is."""
    if _looks_like_input_file(text):
        from .pwinput import parse_pw_input  # imported here: pwinput builds on this module

        return parse_pw_input(text)
    return parse_pw_output(text)


def read_qe_file(path: str | Path) -> QEResult:
    """Read a pw.x input or output file (plain text or gzip) from disk."""
    return parse_qe_file(decode_output(Path(path).read_bytes()))


def read_pw_output(path: str | Path) -> QEResult:
    """Parse a pw.x output file from disk (plain text or .gz)."""
    return parse_pw_output(decode_output(Path(path).read_bytes()))
