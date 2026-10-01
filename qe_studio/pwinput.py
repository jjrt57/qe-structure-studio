"""Reader for Quantum ESPRESSO pw.x input files.

The cell comes from ibrav with celldm(1..6) or A, B, C, cosAB, cosAC, cosBC (all lattice
types, following QE's latgen.f90 and the pw.x input documentation), or from CELL_PARAMETERS
when ibrav = 0. ATOMIC_POSITIONS are converted from alat, bohr, angstrom or crystal units,
and coordinates may be simple arithmetic expressions such as 1/3, as pw.x allows.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from pathlib import Path

import numpy as np

from .parser import BOHR_TO_ANGSTROM, Frame, QEParseError, QEResult, element_from_label

CARDS = ("ATOMIC_SPECIES", "ATOMIC_POSITIONS", "K_POINTS", "CELL_PARAMETERS", "OCCUPATIONS",
         "CONSTRAINTS", "ATOMIC_VELOCITIES", "ATOMIC_FORCES", "ADDITIONAL_K_POINTS", "SOLVENTS",
         "HUBBARD")

IBRAV_NAMES = {
    0: "free lattice from CELL_PARAMETERS",
    1: "cubic P (simple cubic)",
    2: "cubic F (fcc)",
    3: "cubic I (bcc)",
    -3: "cubic I (bcc), symmetric axes",
    4: "hexagonal and trigonal P",
    5: "trigonal R, 3-fold axis c",
    -5: "trigonal R, 3-fold axis <111>",
    6: "tetragonal P",
    7: "tetragonal I (bct)",
    8: "orthorhombic P",
    9: "orthorhombic base-centred (C)",
    -9: "orthorhombic base-centred (C), alternate axes",
    91: "orthorhombic one-face base-centred (A)",
    10: "orthorhombic face-centred",
    11: "orthorhombic body-centred",
    12: "monoclinic P, unique axis c",
    -12: "monoclinic P, unique axis b",
    13: "monoclinic base-centred, unique axis c",
    -13: "monoclinic base-centred, unique axis b",
    14: "triclinic",
}

CLOSE_CONTACT = 0.5  # Å; closer atoms almost always mean wrong position units
_FORTRAN_FLOAT = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")
_RE_KEY = re.compile(r"([A-Za-z_][\w%]*(?:\s*\([^)=]*\))?)\s*=")
_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow}


# ----------------------------------------------------------------------------- lattice

def lattice_from_ibrav(ibrav: int, celldm) -> np.ndarray:
    """Lattice vectors (rows, bohr) for a Bravais-lattice index, as pw.x builds them."""
    a, b_a, c_a, cos4, cos5, cos6 = (float(x) for x in celldm)
    if a <= 0:
        raise QEParseError("celldm(1) (or A) must be positive.")

    def need(value: float, name: str, positive: bool = True) -> float:
        if (positive and value <= 0) or (not positive and not -1 < value < 1):
            raise QEParseError(f"ibrav = {ibrav} needs a valid {name} (got {value}).")
        return value

    if ibrav == 1:
        v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    elif ibrav == 2:
        v = [[-0.5, 0, 0.5], [0, 0.5, 0.5], [-0.5, 0.5, 0]]
    elif ibrav == 3:
        v = [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5]]
    elif ibrav == -3:
        v = [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]
    elif ibrav == 4:
        c = need(c_a, "celldm(3) = c/a")
        v = [[1, 0, 0], [-0.5, math.sqrt(3) / 2, 0], [0, 0, c]]
    elif ibrav in (5, -5):
        cg = cos4
        if not -0.5 < cg < 1:
            raise QEParseError(f"ibrav = {ibrav} needs -0.5 < celldm(4) = cos(gamma) < 1 (got {cg}).")
        tx, ty, tz = math.sqrt((1 - cg) / 2), math.sqrt((1 - cg) / 6), math.sqrt((1 + 2 * cg) / 3)
        if ibrav == 5:
            v = [[tx, -ty, tz], [0, 2 * ty, tz], [-tx, -ty, tz]]
        else:
            u, w = tz - 2 * math.sqrt(2) * ty, tz + math.sqrt(2) * ty
            v = (np.array([[u, w, w], [w, u, w], [w, w, u]]) / math.sqrt(3)).tolist()
    elif ibrav == 6:
        c = need(c_a, "celldm(3) = c/a")
        v = [[1, 0, 0], [0, 1, 0], [0, 0, c]]
    elif ibrav == 7:
        c = need(c_a, "celldm(3) = c/a")
        v = [[0.5, -0.5, c / 2], [0.5, 0.5, c / 2], [-0.5, -0.5, c / 2]]
    elif ibrav in (8, 9, -9, 91, 10, 11):
        b, c = need(b_a, "celldm(2) = b/a"), need(c_a, "celldm(3) = c/a")
        v = {
            8: [[1, 0, 0], [0, b, 0], [0, 0, c]],
            9: [[0.5, b / 2, 0], [-0.5, b / 2, 0], [0, 0, c]],
            -9: [[0.5, -b / 2, 0], [0.5, b / 2, 0], [0, 0, c]],
            91: [[1, 0, 0], [0, b / 2, -c / 2], [0, b / 2, c / 2]],
            10: [[0.5, 0, c / 2], [0.5, b / 2, 0], [0, b / 2, c / 2]],
            11: [[0.5, b / 2, c / 2], [-0.5, b / 2, c / 2], [-0.5, -b / 2, c / 2]],
        }[ibrav]
    elif ibrav in (12, 13):
        b, c = need(b_a, "celldm(2) = b/a"), need(c_a, "celldm(3) = c/a")
        cg = need(cos4, "celldm(4) = cos(ab)", positive=False)
        sg = math.sqrt(1 - cg * cg)
        if ibrav == 12:
            v = [[1, 0, 0], [b * cg, b * sg, 0], [0, 0, c]]
        else:
            v = [[0.5, 0, -c / 2], [b * cg, b * sg, 0], [0.5, 0, c / 2]]
    elif ibrav in (-12, -13):
        b, c = need(b_a, "celldm(2) = b/a"), need(c_a, "celldm(3) = c/a")
        cb = need(cos5, "celldm(5) = cos(ac)", positive=False)
        sb = math.sqrt(1 - cb * cb)
        if ibrav == -12:
            v = [[1, 0, 0], [0, b, 0], [c * cb, 0, c * sb]]
        else:
            v = [[0.5, b / 2, 0], [-0.5, b / 2, 0], [c * cb, 0, c * sb]]
    elif ibrav == 14:
        b, c = need(b_a, "celldm(2) = b/a"), need(c_a, "celldm(3) = c/a")
        ca = need(cos4, "celldm(4) = cos(bc)", positive=False)
        cb = need(cos5, "celldm(5) = cos(ac)", positive=False)
        cg = need(cos6, "celldm(6) = cos(ab)", positive=False)
        sg = math.sqrt(1 - cg * cg)
        term = 1 + 2 * ca * cb * cg - ca * ca - cb * cb - cg * cg
        if term <= 0:
            raise QEParseError("The angles in celldm(4..6) don't form a valid triclinic cell.")
        v = [[1, 0, 0], [b * cg, b * sg, 0], [c * cb, c * (ca - cb * cg) / sg, c * math.sqrt(term) / sg]]
    else:
        raise QEParseError(f"ibrav = {ibrav} is not a Bravais-lattice index pw.x knows.")
    return np.array(v, dtype=float) * a


def celldm_from_abc(ibrav: int, A: float, B: float, C: float,
                    cos_ab: float, cos_ac: float, cos_bc: float) -> list[float]:
    """Convert A, B, C (Å) and cosines to celldm, as QE's abc2celldm does."""
    if A <= 0:
        raise QEParseError("A must be positive.")
    celldm = [A / BOHR_TO_ANGSTROM, B / A, C / A, 0.0, 0.0, 0.0]
    if ibrav in (0, 14):
        celldm[3], celldm[4], celldm[5] = cos_bc, cos_ac, cos_ab
    elif ibrav in (-12, -13):
        celldm[4] = cos_ac
    else:
        celldm[3] = cos_ab
    return celldm


# ----------------------------------------------------------------------------- text parsing

def _strip_comment(line: str, markers: str = "!") -> str:
    quote = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char in markers:
            return line[:index]
    return line


def _namelist_end(text: str) -> tuple[bool, str]:
    """Find a closing '/' (or &END) outside quotes."""
    if re.match(r"\s*&end\b", text, re.I):
        return True, ""
    quote = None
    for index, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "/":
            return True, text[:index]
    return False, text


def _value(raw: str):
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    low = text.lower()
    if low in (".true.", ".t.", "t", "true"):
        return True
    if low in (".false.", ".f.", "f", "false"):
        return False
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if _FORTRAN_FLOAT.fullmatch(text):
        return float(low.replace("d", "e"))
    return text


def _parse_namelist_body(body: str) -> list[tuple[str, str]]:
    strings: list[str] = []

    def hide(match):
        strings.append(match.group(0))
        return f"\x00{len(strings) - 1}\x00"

    masked = re.sub(r"'[^']*'|\"[^\"]*\"", hide, body)
    matches = list(_RE_KEY.finditer(masked))
    entries = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(masked)
        raw = masked[match.end():end].strip().strip(",").strip()
        raw = re.sub(r"\x00(\d+)\x00", lambda m: strings[int(m.group(1))], raw)
        entries.append((re.sub(r"\s+", "", match.group(1)).lower(), raw))
    return entries


def _split_input(text: str):
    """Return ({namelist: [(key, raw value)]}, [cards])."""
    namelists: dict[str, list[tuple[str, str]]] = {}
    cards: list[dict] = []
    current, body = None, []
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line)
        if current is None:
            stripped = line.strip()
            if stripped.startswith("&"):
                match = re.match(r"&(\w+)(.*)", stripped)
                if not match or match.group(1).lower() == "end":
                    continue
                current, body = match.group(1).lower(), []
                ended, before = _namelist_end(match.group(2))
                body.append(before)
                if ended:
                    namelists[current] = _parse_namelist_body("\n".join(body))
                    current = None
                continue
            card_line = _strip_comment(raw_line, "!#").strip()
            if not card_line:
                continue
            head = re.split(r"[\s{(]", card_line, maxsplit=1)[0].upper()
            if head in CARDS:
                option_words = re.sub(r"[{}()=]", " ", card_line[len(head):]).split()
                cards.append({"name": head, "option": option_words[0].lower() if option_words else "",
                              "header": card_line, "lines": []})
            elif cards:
                cards[-1]["lines"].append(card_line)
        else:
            opener = re.match(r"\s*&(\w+)", line)
            if opener and opener.group(1).lower() != "end":
                raise QEParseError(f"The &{current.upper()} namelist is never closed with '/' before "
                                   f"&{opener.group(1).upper()}.")
            ended, before = _namelist_end(line)
            body.append(before)
            if ended:
                namelists[current] = _parse_namelist_body("\n".join(body))
                current = None
    if current is not None:
        raise QEParseError(f"The &{current.upper()} namelist is never closed with '/'.")
    return namelists, cards


def _eval_number(text: str) -> float:
    """A number or a simple arithmetic expression (1/3, 0.5+0.25, sqrt(3)/2, 2^0.5)."""
    text = text.strip()
    if _FORTRAN_FLOAT.fullmatch(text):
        return float(text.lower().replace("d", "e"))
    if len(text) > 64:
        raise QEParseError(f"'{text[:40]}…' is not a number.")
    expression = re.sub(r"(\d)[dD]([-+]?\d)", r"\1e\2", text).replace("^", "**")
    try:
        tree = ast.parse(expression, mode="eval").body
    except SyntaxError:
        raise QEParseError(f"'{text}' is not a number or a simple arithmetic expression.") from None

    def evaluate(node) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 8 or abs(left) > 1e6):
                raise QEParseError(f"'{text}' is too large an expression.")
            try:
                return _OPERATORS[type(node.op)](left, right)
            except ZeroDivisionError:
                raise QEParseError(f"'{text}' divides by zero.") from None
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id.lower() == "sqrt" and len(node.args) == 1 and not node.keywords):
            value = evaluate(node.args[0])
            if value < 0:
                raise QEParseError(f"'{text}' takes the square root of a negative number.")
            return math.sqrt(value)
        raise QEParseError(f"'{text}' is not a number or a simple arithmetic expression.")

    return evaluate(tree)


def _element_for_species(label: str, pseudo: str) -> str:
    try:
        return element_from_label(label)
    except QEParseError:
        match = re.match(r"([A-Za-z]{1,2})", Path(pseudo).name)
        if match:
            for candidate in (match.group(1), match.group(1)[:1]):
                try:
                    return element_from_label(candidate)
                except QEParseError:
                    continue
    raise QEParseError(f"Can't tell which element the species '{label}' is, from its label or its "
                       f"pseudopotential '{pseudo}'.")


def _close_contacts(structure, threshold: float = CLOSE_CONTACT) -> list[tuple[int, int, float]]:
    if len(structure) < 2 and min(structure.lattice.abc) >= threshold:
        return []
    if len(structure) <= 1500:
        distances = structure.distance_matrix
        np.fill_diagonal(distances, np.inf)
        i_index, j_index = np.where(np.triu(distances < threshold, 1))
        pairs = [(int(i), int(j), float(distances[i, j])) for i, j in zip(i_index, j_index)]
    else:
        centres, neighbours, _, dists = structure.get_neighbor_list(threshold)
        pairs = sorted({(int(min(i, j)), int(max(i, j)), round(float(d), 4))
                        for i, j, d in zip(centres, neighbours, dists) if i != j})
    return sorted(pairs, key=lambda pair: pair[2])


# ----------------------------------------------------------------------------- main entry

def parse_pw_input(text: str) -> QEResult:
    """Parse the text of a pw.x input file into a single-geometry QEResult."""
    namelists, cards = _split_input(text)
    if "system" not in namelists:
        raise QEParseError("There is no &SYSTEM namelist, so this isn't a pw.x input file.")
    system = {key: _value(raw) for key, raw in namelists["system"]}
    control = {key: _value(raw) for key, raw in namelists.get("control", [])}
    first_card = {}
    for card in cards:
        first_card.setdefault(card["name"], card)
    notes: list[tuple[str, str]] = []

    if system.get("space_group"):
        raise QEParseError("This input uses space_group with crystal_sg positions, which QE Structure "
                           "Studio can't expand yet. Upload the pw.x output instead: it lists every atom.")
    if "ibrav" not in system:
        raise QEParseError("ibrav is not set in &SYSTEM.")
    try:
        ibrav = int(system["ibrav"])
    except (TypeError, ValueError):
        raise QEParseError(f"ibrav = {system['ibrav']} is not an integer.") from None

    celldm = [0.0] * 6
    if isinstance(system.get("celldm"), str):  # celldm = 10.2, 0, 1.6 (array form)
        for index, number in enumerate(_FORTRAN_FLOAT.findall(system["celldm"])[:6]):
            celldm[index] = float(number.lower().replace("d", "e"))
    for index in range(6):
        value = system.get(f"celldm({index + 1})")
        if value is not None:
            try:
                celldm[index] = float(value)
            except (TypeError, ValueError):
                raise QEParseError(f"celldm({index + 1}) = {value} is not a number.") from None
    abc_keys = [key for key in ("a", "b", "c", "cosab", "cosac", "cosbc") if key in system]
    lattice_source = "celldm"
    if abc_keys:
        if celldm[0]:
            notes.append(("warning", "Both celldm and A, B, C are set, which pw.x rejects. celldm is used here."))
        else:
            try:
                numbers = {key: float(system.get(key, 0.0) or 0.0) for key in ("a", "b", "c", "cosab", "cosac", "cosbc")}
            except (TypeError, ValueError):
                raise QEParseError("A, B, C and cosAB, cosAC, cosBC must be numbers.") from None
            celldm = celldm_from_abc(ibrav, numbers["a"], numbers["b"], numbers["c"],
                                     numbers["cosab"], numbers["cosac"], numbers["cosbc"])
            lattice_source = "A, B, C"
    alat = celldm[0] or None

    cell_card = first_card.get("CELL_PARAMETERS")
    cell_units = None
    if ibrav == 0:
        if cell_card is None:
            raise QEParseError("ibrav = 0 needs a CELL_PARAMETERS card.")
        if len(cell_card["lines"]) < 3:
            raise QEParseError("CELL_PARAMETERS needs three lattice vectors.")
        try:
            vectors = np.array([[_eval_number(x) for x in line.split()[:3]] for line in cell_card["lines"][:3]])
        except QEParseError as exc:
            raise QEParseError(f"CELL_PARAMETERS: {exc}") from None
        if vectors.shape != (3, 3):
            raise QEParseError("Each CELL_PARAMETERS line needs three numbers.")
        option = cell_card["option"]
        if option == "alat" or (option == "" and alat):
            if not alat:
                raise QEParseError("CELL_PARAMETERS alat needs celldm(1) or A to set the lattice parameter.")
            lattice_bohr = vectors * alat
            cell_units = "alat"
            if option == "":
                notes.append(("info", "CELL_PARAMETERS has no units, so pw.x reads it in units of the "
                                      "lattice parameter (deprecated; write CELL_PARAMETERS alat)."))
        elif option in ("bohr", "angstrom", ""):
            lattice_bohr = vectors if option in ("bohr", "") else vectors / BOHR_TO_ANGSTROM
            if option == "":
                notes.append(("info", "CELL_PARAMETERS has no units and no lattice parameter is set, so "
                                      "pw.x reads it in bohr (deprecated; write CELL_PARAMETERS bohr)."))
            elif alat:
                notes.append(("warning", f"celldm(1) or A and CELL_PARAMETERS {option} both set the lattice "
                                         "parameter, which pw.x rejects. The cell is taken from CELL_PARAMETERS."))
            cell_units = option or "bohr"
            alat = float(np.linalg.norm(lattice_bohr[0]))
            lattice_source = "CELL_PARAMETERS"
        else:
            raise QEParseError(f"Unknown CELL_PARAMETERS units '{option}'.")
    else:
        if ibrav not in IBRAV_NAMES:
            raise QEParseError(f"ibrav = {ibrav} is not a Bravais-lattice index pw.x knows.")
        if not alat:
            raise QEParseError(f"ibrav = {ibrav} needs celldm(1) or A to set the lattice parameter.")
        lattice_bohr = lattice_from_ibrav(ibrav, celldm)
        if cell_card is not None:
            notes.append(("info", f"CELL_PARAMETERS is ignored because ibrav = {ibrav}."))
        if ibrav == -13:
            notes.append(("info", "ibrav = -13 is read with the axes used since QE 6.4.1. Older versions "
                                  "defined a1 and a2 differently."))
    if abs(np.linalg.det(lattice_bohr)) < 1e-6:
        raise QEParseError("The lattice vectors are coplanar, so they don't define a cell.")

    positions_card = first_card.get("ATOMIC_POSITIONS")
    if positions_card is None:
        raise QEParseError("There is no ATOMIC_POSITIONS card.")
    position_units = positions_card["option"]
    if position_units == "crystal_sg":
        raise QEParseError("crystal_sg positions list only symmetry-unique atoms, which QE Structure Studio "
                           "can't expand yet. Upload the pw.x output instead: it lists every atom.")
    nat = system.get("nat")
    lines = positions_card["lines"]
    if nat is None:
        nat = len(lines)
        notes.append(("warning", f"nat is not set in &SYSTEM; read {nat} atoms from ATOMIC_POSITIONS."))
    nat = int(nat)
    if len(lines) < nat:
        raise QEParseError(f"ATOMIC_POSITIONS lists {len(lines)} atoms, but nat = {nat}.")
    if len(lines) > nat:
        notes.append(("warning", f"ATOMIC_POSITIONS has {len(lines) - nat} more line(s) than nat = {nat}; "
                                 "pw.x stops on them, and they were ignored here."))
    labels, coords, if_pos = [], [], []
    for line in lines[:nat]:
        fields = line.split()
        if len(fields) not in (4, 7):
            raise QEParseError(f"Can't read the ATOMIC_POSITIONS line '{line}'. Each line needs a label, "
                               "three coordinates and optionally three 0/1 flags.")
        labels.append(fields[0])
        coords.append([_eval_number(x) for x in fields[1:4]])
        if_pos.append([int(float(x)) for x in fields[4:7]] if len(fields) == 7 else [1, 1, 1])
    coords = np.array(coords, dtype=float)
    inverse = np.linalg.inv(lattice_bohr)
    if position_units in ("", "alat"):
        if position_units == "":
            notes.append(("info", "ATOMIC_POSITIONS has no units, so pw.x assumes alat (deprecated)."))
        frac = (coords * alat) @ inverse
    elif position_units == "bohr":
        frac = coords @ inverse
    elif position_units == "angstrom":
        frac = (coords / BOHR_TO_ANGSTROM) @ inverse
    elif position_units == "crystal":
        frac = coords
        if np.abs(coords).max(initial=0) > 2:
            notes.append(("warning", "Some crystal coordinates are larger than 2. If they are Cartesian, "
                                     "change the ATOMIC_POSITIONS units."))
    else:
        raise QEParseError(f"Unknown ATOMIC_POSITIONS units '{position_units}'.")

    species = []
    element_map: dict[str, str] = {}
    species_card = first_card.get("ATOMIC_SPECIES")
    ntyp = system.get("ntyp")
    if species_card is None:
        notes.append(("warning", "There is no ATOMIC_SPECIES card; elements are taken from the atom labels."))
    else:
        species_lines = species_card["lines"][: int(ntyp)] if ntyp else species_card["lines"]
        for line in species_lines:
            fields = line.split()
            if len(fields) < 3:
                raise QEParseError(f"Can't read the ATOMIC_SPECIES line '{line}'. It needs a label, a mass "
                                   "and a pseudopotential file.")
            element = _element_for_species(fields[0], fields[2])
            species.append({"label": fields[0], "mass": fields[1], "pseudo": fields[2], "element": element})
            element_map[fields[0]] = element
        if ntyp is not None and len(species) != int(ntyp):
            notes.append(("warning", f"ntyp = {ntyp} but ATOMIC_SPECIES lists {len(species)} species."))
        missing = sorted(set(labels) - set(element_map))
        if missing:
            notes.append(("warning", f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} used in "
                                     "ATOMIC_POSITIONS but missing from ATOMIC_SPECIES; pw.x will stop."))
    for label in labels:
        if label not in element_map:
            element_map[label] = element_from_label(label)

    frame = Frame(lattice=lattice_bohr * BOHR_TO_ANGSTROM, labels=labels, frac_coords=np.array(frac),
                  kind="input", is_final=True, species_elements=element_map)
    structure = frame.structure()
    if min(structure.lattice.abc) < CLOSE_CONTACT:
        notes.append(("warning", f"A lattice vector is only {min(structure.lattice.abc):.3f} Å long. "
                                 "Check the units of the lattice parameter."))
    contacts = _close_contacts(structure)
    if contacts:
        shown = "; ".join(f"atoms {i + 1} ({labels[i]}) and {j + 1} ({labels[j]}) are {d:.3f} Å apart"
                          for i, j, d in contacts[:3])
        more = f", and {len(contacts) - 3} more pairs" if len(contacts) > 3 else ""
        notes.append(("warning", f"Atoms overlap: {shown}{more}. This usually means ATOMIC_POSITIONS "
                                 f"is in the wrong units (it says '{position_units or 'alat'}')."))
    fixed = sum(1 for flags in if_pos if flags != [1, 1, 1])
    if fixed:
        notes.append(("info", f"{fixed} atom{'s have' if fixed > 1 else ' has'} fixed coordinates (if_pos). "
                              "They are kept in the Quantum ESPRESSO input download."))

    kpoints = "not set"
    k_card = first_card.get("K_POINTS")
    if k_card is not None:
        option = k_card["option"] or "tpiba"
        if option == "automatic" and k_card["lines"]:
            numbers = k_card["lines"][0].split()
            kpoints = f"{'×'.join(numbers[:3])} grid" + (f", shift {' '.join(numbers[3:6])}" if len(numbers) >= 6 else "")
        elif option == "gamma":
            kpoints = "Γ point only"
        elif k_card["lines"]:
            kpoints = f"{k_card['lines'][0].split()[0]} points ({option})"

    input_data = {
        "namelists": namelists,
        "cards": cards,
        "species": species,
        "if_pos": if_pos,
        "ibrav": ibrav,
        "ibrav_name": IBRAV_NAMES.get(ibrav, ""),
        "celldm": celldm,
        "lattice_source": lattice_source,
        "kpoints": kpoints,
        "ecutwfc": system.get("ecutwfc"),
        "ecutrho": system.get("ecutrho"),
        "nat": nat,
        "ntyp": ntyp,
    }
    return QEResult(
        frames=[frame],
        calculation=str(control.get("calculation", "scf")).lower(),
        converged=None,
        job_done=False,
        alat_bohr=alat,
        cell_units=cell_units,
        position_units=position_units or "alat",
        notes=notes,
        source="input",
        input_data=input_data,
    )
