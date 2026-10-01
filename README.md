# QE Structure Studio

Read a Quantum ESPRESSO `pw.x` input or output file, check the structure in an interactive 3D
view, and download it as CIF, XYZ, POSCAR or nine other formats. It runs as a Streamlit web app and as a
command-line converter.

![QE Structure Studio showing a relaxed rutile TiO2 cell](docs/screenshot.png)

## What it does

- Reads scf, relax, vc-relax and md outputs, whatever units pw.x printed: `alat`, `bohr`,
  `angstrom` or `crystal`.
- Reads pw.x input files too: every `ibrav` (1 to 14, including −3, −5, −9, 91, −12 and −13)
  with `celldm` or `A, B, C, cosAB, cosAC, cosBC`, and `ibrav = 0` with `CELL_PARAMETERS` in
  alat, bohr or angstrom. All `ATOMIC_POSITIONS` units work, as do coordinate expressions such
  as `1/3` and `if_pos` flags. The geometry was checked against what pw.x 6.7 itself prints for
  36 different inputs.
- Checks input files for common mistakes: overlapping atoms (usually wrong position units),
  `nat` or `ntyp` mismatches, species missing from `ATOMIC_SPECIES` and unclosed namelists.
- Keeps every ionic step. A slider lets you view and export any step, not only the final one.
- Shows the structure in 3D (3Dmol.js). Atoms on cell faces and corners are repeated as in VESTA,
  the unit cell is drawn with coloured a/b/c axes, and you can pick supercells, three drawing
  styles and VESTA or Jmol colours. Hover an atom for its fractional coordinates; save a PNG.
- Reports the space group (spglib), Wyckoff positions, lattice parameters, volume, density and
  total energy.
- Can switch the cell to the conventional or primitive standard setting for both the view
  and the downloads.
- Plots energy, forces, pressure and volume across the relaxation, and exports the whole
  trajectory as extended XYZ.
- Flags runs that did not finish, relaxations that hit `nstep`, SCF cycles that did not converge
  and large final-SCF energy changes (Pulay stress).

## Export formats

| Format | File | Notes |
| --- | --- | --- |
| CIF | `.cif` | Every atom listed (P1) |
| CIF, symmetrized | `_sym.cif` | Space group from spglib, symmetry-unique atoms only |
| XYZ | `.xyz` | Cartesian Å, no cell |
| Extended XYZ | `.extxyz` | Lattice, periodicity and energy (ASE, OVITO) |
| VASP POSCAR | `.vasp` | Rename to `POSCAR` for VASP |
| Quantum ESPRESSO input | `.in` | `ibrav = 0`. From an input file it keeps all your settings; from an output, pseudopotentials, cutoff and k-points are placeholders |
| XCrySDen XSF | `.xsf` | |
| PDB | `.pdb` | With a CRYST1 record |
| LAMMPS data | `.lmp` | `atom_style atomic`, metal units, masses |
| CASTEP cell | `.cell` | |
| FHI-aims geometry | `_geometry.in` | |
| pymatgen JSON | `.json` | Materials Project format |

"Download every format" puts all twelve in one zip file.

## Run it locally

Python 3.11 or newer (3.12 recommended):

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501 with three bundled examples.

## Put it online with GitHub and Streamlit Community Cloud

1. Create an empty repository on GitHub, for example `qe-structure-studio`.
2. Push this folder to it:
   ```bash
   git init
   git add .
   git commit -m "QE Structure Studio"
   git branch -M main
   git remote add origin https://github.com/<your-username>/qe-structure-studio.git
   git push -u origin main
   ```
3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
4. Click **Create app** and choose to deploy from GitHub. Pick your repository, the `main`
   branch and `app.py` as the main file. Under advanced settings choose Python 3.12. Pick a
   subdomain if you like.
5. Click **Deploy**. The first build installs pymatgen and takes a few minutes. The app is then
   live at `https://<your-subdomain>.streamlit.app`, and every push to `main` redeploys it.

Optionally set `GITHUB_URL` near the top of `app.py` to show a link to your repository in the
footer. The GitHub Actions workflow in `.github/workflows/tests.yml` runs the tests on every push.

## Run it with GitHub only (Codespaces)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/YOUR-USERNAME/qe-structure-studio?quickstart=1)

GitHub Pages can't host this app, because it only serves static files and the app needs Python
running on a server. GitHub Codespaces can run it, though, and the repository includes a dev
container that sets everything up:

1. On your repository page click **Code → Codespaces → Create codespace on main**, or use the
   badge above after replacing `YOUR-USERNAME`.
2. The first start takes a few minutes while the requirements install. The app then starts by
   itself and opens in a preview tab.
3. To share it, open the **Ports** tab, right-click port 8501 and choose
   **Port visibility → Public**. Anyone with that `….app.github.dev` link can use the app while
   your codespace is running.

A codespace stops after 30 minutes of inactivity by default, and running it uses your monthly
free Codespaces hours. That makes it good for your own use and for demos. For a permanent public
site, use Streamlit Community Cloud as described above.

## Command line

`qe2cif.py` keeps the original interface (`-i`, `-o` and the `convert_qe_to_cif()` function). The
output extension picks the format:

```bash
python qe2cif.py -i vc-relax.out -o relaxed.cif
python qe2cif.py -i vc-relax.out -o relaxed.cif --symmetrize --symprec 0.01
python qe2cif.py -i relax.out -o POSCAR.vasp --step 0        # the input geometry
python qe2cif.py -i md.out -o trajectory.extxyz --all-steps
python qe2cif.py -i zno.in -o zno.cif                        # input files work too
python qe2cif.py -i zno.in -o zno_ibrav0.in                  # same input, cell written out
```

It exits with status 1 on errors and prints parser warnings to stderr, so it can be used in
scripts and workflows.

## Changes from the original script

The original `convert_qe_to_cif` had the right idea (the last geometry block in the file is the
final structure), but it failed or silently gave wrong results in common cases:

- **Units were ignored.** For `ibrav ≠ 0` runs pw.x prints `CELL_PARAMETERS (alat= …)` and
  `ATOMIC_POSITIONS (alat)`. The script read those numbers as Å. On the bundled silicon example
  it reports a = 0.7281 Å instead of 3.8144 Å. `bohr` output was off by a factor of 0.529.
- **Fixed-cell relax and scf outputs always failed.** They contain no `CELL_PARAMETERS` block; the
  cell is only in the header. The parser now reads the header geometry (`crystal axes` and the
  `tau` table).
- **Species labels became ions.** pymatgen reads the QE label `Fe1` as Fe⁺ and `Fe2` as Fe²⁺,
  which wrote wrong oxidation states into the CIF, and `Fe_up` crashed. Labels are now mapped to
  elements and kept as the `qe_label` site property; the pw.x input export writes them back.
- **Errors were silent.** Exceptions were printed with exit status 0, and a relaxation that hit
  `nstep` returned the next BFGS-proposed geometry, which pw.x never computed, as if it were
  relaxed. Both cases are now reported.
- Numbers in fixed-width Fortran fields can touch (`-100.123456789-12.3…`); they are now split
  correctly.

## Reading input files

The input reader builds the cell exactly as pw.x does (QE's `latgen` conventions, with the
`ibrav = -13` axes used since QE 6.4.1). The "Quantum ESPRESSO input" download of an input file
is the same input with `ibrav = 0`: every namelist variable, pseudopotential, k-point setting,
constraint and extra card is kept, and only the cell and positions are written out explicitly.
Many tools, including ASE and phonopy, need this form. `cell_dofree = 'ibrav'` becomes `'all'`,
because it only works with `ibrav ≠ 0`.

Inputs that use `space_group` with `crystal_sg` positions list only the symmetry-unique atoms
and are not supported yet. Upload the pw.x output of such a run instead, which lists every atom.

## How units are converted

| Block | Units in the file | Conversion |
| --- | --- | --- |
| `CELL_PARAMETERS` | `alat= A` | × A × 0.529177210903 Å |
| | `bohr` | × 0.529177210903 Å |
| | `angstrom` | as is |
| `ATOMIC_POSITIONS` | `alat` | × alat × 0.529177210903 Å, Cartesian |
| | `bohr` | × 0.529177210903 Å, Cartesian |
| | `angstrom` | Cartesian Å |
| | `crystal` | fractional |
| Header `crystal axes` and `tau` | alat | × alat (from `celldm(1)`) |

## Project layout

```
app.py                  Streamlit app
qe2cif.py               command-line converter
qe_studio/parser.py     pw.x output parser
qe_studio/pwinput.py    pw.x input reader (all ibrav lattices)
qe_studio/exporters.py  file formats
qe_studio/viewer.py     3D viewer page (3Dmol.js)
qe_studio/analysis.py   symmetry and cell settings
examples/               example outputs and the inputs that produced them
tests/                  pytest suite and extra real outputs
.streamlit/config.toml  theme and upload limit
.devcontainer/          GitHub Codespaces setup
```

## Tests

```bash
pip install pytest
pytest
```

The tests use real pw.x outputs, and for input files a reference set of 36 inputs covering every
`ibrav` and unit variant, with the geometry pw.x 6.7 printed for each. They check the parser against ASE's independent reader,
cover every unit system, relax/vc-relax/scf/md runs, runs stopped at `nstep`, truncated files,
species labels and gzip input, and read every export format back with pymatgen or ASE.

## About the example data

The outputs in `examples/` and `tests/data/` were produced with Quantum ESPRESSO 6.7 using LDA
pseudopotentials from the QE distribution and deliberately low cutoffs, so they run in seconds.
They are for demonstration, not production results. The inputs are in `examples/inputs/`.
`examples/zno_wurtzite.in` is an input file at the experimental wurtzite lattice.
This is also why the rutile example shows a Pulay-stress note: its final SCF differs from the
last BFGS step because the cutoff is low.
