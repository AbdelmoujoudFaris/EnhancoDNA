# Desktop GUI

```bash
enhancoai gui
```

A PySide6 application (`enhancoai.app`) with a shared, mutable
`ProjectState` (`enhancoai.app.state.ProjectState`) passed to every tab so
work in one tab (e.g. loading a structure) is immediately usable in the
next (e.g. running contact analysis).

## Tabs

- **Project** -- dashboard: project/structure/trajectory summary and
  headline metrics (cooperativity score, interface residue counts).
- **Structure** -- load a PDB/mmCIF file, view the automatic chain
  classification table, a missing-backbone-atom report, and a matplotlib
  3D scatter of the structure (PyVista used instead when installed).
- **Interactions** -- pick a protein chain and a DNA chain, run
  protein-DNA contact analysis, view the contact-map heatmap.
- **MD** -- load a topology (+ optional trajectory), compute/plot RMSD
  and RMSF.
- **Cooperativity** -- enter raw evidence measurements (energetic shift,
  dynamic shift, orientation shift, DNA pathway strength, interface
  persistence) and see the transparent, decomposable Cooperativity Score
  breakdown.
- **AI** -- device info, model/checkpoint selection, and a one-click
  cooperativity prediction on the loaded structure's interface (reports
  "Model weights unavailable" rather than a number when no checkpoint is
  trained).
- **Visualisation** -- render mode selector (Protein / DNA / Protein-DNA
  contacts / Allosteric network).
- **Hypothesis Testing** -- the evidence-linked YES/NO/UNCERTAIN dashboard
  (section 55); re-evaluates from whatever has actually been computed in
  the current session, never fabricates an answer.

## Headless / CI usage

The GUI can be constructed and driven without a display for smoke tests:

```bash
QT_QPA_PLATFORM=offscreen python -c "
from PySide6.QtWidgets import QApplication
from enhancoai.app.main_window import MainWindow
app = QApplication([])
win = MainWindow()
win.structure_panel._load_from_path('data/examples/demo_tf_dna_two_factor.pdb')
"
```
