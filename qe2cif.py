#!/usr/bin/env python3
"""Convert a Quantum ESPRESSO pw.x output or input file to CIF or another structure format.

A drop-in replacement for the original script: the same -i/-o options and the
same convert_qe_to_cif() function, now unit-aware (alat, bohr, angstrom,
crystal), able to read relax/scf/md outputs, and exiting with status 1 on errors.

Examples:
    python qe2cif.py -i vc-relax.out -o relaxed.cif
    python qe2cif.py -i vc-relax.out -o relaxed.cif --symmetrize
    python qe2cif.py -i relax.out -o POSCAR.vasp --step 0      # the input geometry
    python qe2cif.py -i md.out -o trajectory.extxyz --all-steps
    python qe2cif.py -i zno.in -o zno.cif                     # input files work too
    python qe2cif.py -i zno.in -o zno_ibrav0.in               # same input, cell written out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qe_studio.exporters import FORMATS, FORMATS_BY_KEY, export_structure, export_trajectory, format_for_filename
from qe_studio.parser import QEParseError, read_qe_file


def convert_qe_to_cif(input_file, output_file, *, step=-1, fmt=None, symmetrize=False,
                      symprec=0.01, wrap=True):
    """Write one geometry from a pw.x output or input file. Returns (result, structure, format)."""
    result = read_qe_file(input_file)
    frame = result.frames[step]
    structure = frame.structure(wrap=wrap)
    export_format = FORMATS_BY_KEY[fmt] if fmt else format_for_filename(str(output_file), symmetrize)
    title = f"{structure.composition.reduced_formula} from {Path(input_file).name}"
    text = export_structure(structure, export_format.key, title=title, energy_ev=frame.energy_ev,
                            symprec=symprec, qe_template=result.input_data if keeps_settings(result) else None)
    Path(output_file).write_text(text)
    return result, structure, export_format


def keeps_settings(result) -> bool:
    """Keep the settings of an input file in the Quantum ESPRESSO export."""
    return result.source == "input"


def main(argv=None) -> int:
    formats = ", ".join(f"{f.key} ({f.suffix})" for f in FORMATS)
    parser = argparse.ArgumentParser(
        description="Convert a Quantum ESPRESSO pw.x output or input file to CIF, XYZ, POSCAR and more.",
        epilog=f"Formats: {formats}")
    parser.add_argument("-i", "--input", required=True, help="pw.x output or input file (plain text or .gz)")
    parser.add_argument("-o", "--output", required=True, help="output file; its extension picks the format")
    parser.add_argument("-f", "--format", choices=list(FORMATS_BY_KEY), help="force a format")
    parser.add_argument("--step", type=int, default=-1,
                        help="ionic step to write: 0 is the input geometry, -1 the final one (default)")
    parser.add_argument("--all-steps", action="store_true",
                        help="write every ionic step as a multi-frame extended XYZ file")
    parser.add_argument("--symmetrize", action="store_true",
                        help="for .cif output, write the symmetrized structure with its space group")
    parser.add_argument("--symprec", type=float, default=0.01, help="symmetry tolerance in Å (default 0.01)")
    parser.add_argument("--no-wrap", action="store_true", help="don't wrap atoms into the unit cell")
    args = parser.parse_args(argv)

    print(f"Parsing Quantum ESPRESSO file: {args.input}...")
    try:
        if args.all_steps:
            result = read_qe_file(args.input)
            frames = result.frames
            text = export_trajectory([f.structure(wrap=not args.no_wrap) for f in frames],
                                     [f.energy_ev for f in frames],
                                     [{"step": i, "pressure_kbar": f.pressure_kbar} for i, f in enumerate(frames)])
            Path(args.output).write_text(text)
            structure, label = frames[-1].structure(), f"{len(frames)} steps as extended XYZ"
        else:
            result, structure, export_format = convert_qe_to_cif(
                args.input, args.output, step=args.step, fmt=args.format, symmetrize=args.symmetrize,
                symprec=args.symprec, wrap=not args.no_wrap)
            label = export_format.label
    except FileNotFoundError:
        print(f"Error: could not find the file '{args.input}'.", file=sys.stderr)
        return 1
    except IndexError:
        print(f"Error: step {args.step} doesn't exist in {args.input}.", file=sys.stderr)
        return 1
    except (QEParseError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for level, message in result.notes:
        print(f"{level.capitalize()}: {message}", file=sys.stderr)
    if result.source == "input":
        data = result.input_data or {}
        print(f"Read the structure from a pw.x input (ibrav = {data.get('ibrav')}, {data.get('ibrav_name', '')}).")
    else:
        status = {True: "converged", False: "NOT converged", None: "convergence not reported"}[result.converged]
        print(f"Read {len(result.frames)} geometries ({result.calculation}, {status}).")
    print(f"Successfully converted {args.input} -> {args.output} ({label})")
    lat = structure.lattice
    print(f"Final Lattice Parameters: a={lat.a:.4f} Å, b={lat.b:.4f} Å, c={lat.c:.4f} Å, "
          f"alpha={lat.alpha:.3f}°, beta={lat.beta:.3f}°, gamma={lat.gamma:.3f}°")
    return 0


if __name__ == "__main__":
    sys.exit(main())
