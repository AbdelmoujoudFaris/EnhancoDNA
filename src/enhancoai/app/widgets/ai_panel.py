"""AI tab (sections 23-34, 41-42, 61): model selection, device, inference."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QFileDialog, QLineEdit, QGroupBox, QTextEdit, QFormLayout,
)

from enhancoai.app.state import ProjectState


class AIPanel(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)

        device_group = QGroupBox("Compute Device")
        device_layout = QHBoxLayout(device_group)
        self.device_label = QLabel("Detecting...")
        device_layout.addWidget(self.device_label)
        layout.addWidget(device_group)
        self._detect_device()

        config_group = QGroupBox("Model")
        config_form = QFormLayout(config_group)
        self.architecture_combo = QComboBox()
        self.architecture_combo.addItems(["hybrid", "cnn3d", "gnn", "temporal"])
        config_form.addRow("Architecture:", self.architecture_combo)

        checkpoint_row = QHBoxLayout()
        self.checkpoint_edit = QLineEdit()
        self.checkpoint_edit.setPlaceholderText("models/pretrained/model.pt")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_checkpoint)
        checkpoint_row.addWidget(self.checkpoint_edit)
        checkpoint_row.addWidget(browse_button)
        config_form.addRow("Checkpoint:", checkpoint_row)
        layout.addWidget(config_group)

        actions = QHBoxLayout()
        self.predict_button = QPushButton("Run Cooperativity Prediction (3D CNN)")
        self.predict_button.clicked.connect(self._predict)
        actions.addWidget(self.predict_button)
        layout.addLayout(actions)

        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setPlaceholderText(
            "Prediction, probability, mechanism, confidence and evidence appear here (section 61 format)."
        )
        layout.addWidget(self.output_box, 1)

    def _detect_device(self) -> None:
        from enhancoai.utils.device import detect_device, describe

        info = detect_device()
        self.device_label.setText(describe(info))

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Model checkpoint", "", "PyTorch checkpoint (*.pt)")
        if path:
            self.checkpoint_edit.setText(path)

    def _predict(self) -> None:
        if self.state.structure is None or not self.state.protein_chains or not self.state.dna_chains:
            self.output_box.setPlainText("Load a structure with at least one protein and one DNA chain first.")
            return

        from enhancoai.voxel.voxelizer import voxelize_interface, to_tensor
        from enhancoai.structure.selection import heavy_atoms
        from enhancoai.inference.predictor import Predictor
        from enhancoai.utils.config import ModelConfig

        protein_atoms = heavy_atoms(self.state.structure.chain(self.state.protein_chains[0]))
        dna_atoms = heavy_atoms(self.state.structure.chain(self.state.dna_chains[0]))
        voxel_grid = voxelize_interface(protein_atoms, dna_atoms)
        voxel_tensor = to_tensor(voxel_grid).float()

        checkpoint_path = self.checkpoint_edit.text().strip() or "models/pretrained/model.pt"
        model_config = ModelConfig(architecture="cnn3d")
        predictor = Predictor(checkpoint_path, model_config)
        prediction = predictor.predict(voxel_grid=voxel_tensor)

        lines = [
            f"Prediction: {'Cooperative' if prediction.is_cooperative else 'Non-cooperative'}",
            f"Probability: {prediction.probability:.3f}" if prediction.probability == prediction.probability else "Probability: n/a",
            f"Confidence: {prediction.confidence:.3f}",
        ]
        if not prediction.weights_available:
            lines = prediction.evidence
        else:
            lines.append("")
            lines.append("Evidence:")
            lines.extend(f"  - {e}" for e in prediction.evidence)
        self.output_box.setPlainText("\n".join(lines))
