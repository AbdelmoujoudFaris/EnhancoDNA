"""EnhancoAI command-line interface (section 50)."""

from __future__ import annotations

import json
from pathlib import Path

import click

from enhancoai import __version__
from enhancoai.utils.logging import get_logger

logger = get_logger(__name__)


@click.group()
@click.version_option(version=__version__, prog_name="enhancoai")
def main() -> None:
    """EnhancoAI: deep learning, MD and AI analysis of TF cooperativity and DNA allostery."""


@main.command()
def gui() -> None:
    """Launch the EnhancoAI desktop application."""
    from enhancoai.app.main import run_app

    run_app()


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="PDB or mmCIF file.")
def structure(input_path: str) -> None:
    """Load a structure and report chain classification."""
    from enhancoai.structure.parser import load_structure
    from enhancoai.structure.chain_detection import summarize_chains
    from enhancoai.structure.cleaning import report_missing_backbone_atoms

    struct = load_structure(input_path)
    click.echo(f"Loaded {input_path}: {struct.n_atoms()} atoms, {struct.n_models()} model(s).")
    click.echo(summarize_chains(struct).to_string(index=False))

    missing = report_missing_backbone_atoms(struct)
    if missing:
        click.echo(f"\n{len(missing)} residue(s) with missing backbone atoms (see docs for details).")


@main.command()
@click.option("--structure", "structure_path", required=True, type=click.Path(exists=True))
@click.option("--protein", "protein_chains", multiple=True, required=True, help="Protein chain id(s).")
@click.option("--dna", "dna_chains", multiple=True, required=True, help="DNA chain id(s).")
@click.option("--cutoff", default=5.0, show_default=True, help="Heavy-atom contact cutoff (A).")
def contacts(structure_path: str, protein_chains: tuple, dna_chains: tuple, cutoff: float) -> None:
    """Detect protein-DNA (and protein-protein, if >1 protein chain) contacts."""
    from enhancoai.structure.parser import load_structure
    from enhancoai.interactions.protein_dna import analyse_protein_dna_interactions, format_contact_descriptions
    from enhancoai.interactions.protein_protein import analyse_protein_protein_interactions

    struct = load_structure(structure_path)
    for p_chain in protein_chains:
        for d_chain in dna_chains:
            result = analyse_protein_dna_interactions(struct, p_chain, d_chain, heavy_atom_cutoff=cutoff)
            click.echo(f"\n=== Protein {p_chain} <-> DNA {d_chain} ===")
            click.echo(json.dumps(result.summary(), indent=2))
            for line in format_contact_descriptions(result)[:20]:
                click.echo(f"  {line}")

    if len(protein_chains) > 1:
        for i, chain_a in enumerate(protein_chains):
            for chain_b in protein_chains[i + 1 :]:
                result = analyse_protein_protein_interactions(struct, chain_a, chain_b, heavy_atom_cutoff=cutoff)
                click.echo(f"\n=== Protein {chain_a} <-> Protein {chain_b} ===")
                click.echo(json.dumps(result.summary(), indent=2))


@main.command()
@click.option("--topology", required=True, type=click.Path(exists=True))
@click.option("--trajectory", type=click.Path(exists=True), default=None)
@click.option("--protein-selection", default="protein and name CA", show_default=True)
@click.option("--dna-selection", default="nucleic", show_default=True)
def md(topology: str, trajectory: str | None, protein_selection: str, dna_selection: str) -> None:
    """Run basic MD trajectory analysis (RMSD, RMSF) on protein and DNA selections."""
    from enhancoai.md.loader import load_trajectory
    from enhancoai.md.rmsd import compute_rmsd
    from enhancoai.md.rmsf import compute_rmsf

    handle = load_trajectory(topology, trajectory)
    click.echo(f"Loaded {handle.n_frames} frame(s).")

    for label, selection in (("protein", protein_selection), ("DNA", dna_selection)):
        try:
            rmsd = compute_rmsd(handle, selection=selection)
            rmsf = compute_rmsf(handle, selection=selection)
            click.echo(f"\n{label} RMSD: mean={rmsd['rmsd'].mean():.3f} A, final={rmsd['rmsd'].iloc[-1]:.3f} A")
            click.echo(f"{label} RMSF: mean={rmsf['rmsf'].mean():.3f} A, max={rmsf['rmsf'].max():.3f} A")
        except ValueError as exc:
            click.echo(f"{label}: {exc}")


@main.command()
@click.option("--system-a", "system_a", required=True, type=click.Path(exists=True), help="Directory with TF-A + DNA reaction-coordinate samples (CSV).")
@click.option("--system-ab", "system_ab", required=True, type=click.Path(exists=True), help="Directory with TF-A + TF-B + DNA reaction-coordinate samples (CSV).")
@click.option("--column", default="com_distance", show_default=True)
@click.option("--temperature", default=300.0, show_default=True)
def cooperativity(system_a: str, system_ab: str, column: str, temperature: float) -> None:
    """Compute PMFs for two systems and the cooperativity free energy (ΔΔG_coop)."""
    import pandas as pd
    from enhancoai.free_energy.pmf import compute_pmf_1d
    from enhancoai.free_energy.cooperativity import cooperativity_free_energy

    samples_a = pd.read_csv(system_a)[column].to_numpy()
    samples_ab = pd.read_csv(system_ab)[column].to_numpy()

    pmf_a = compute_pmf_1d(samples_a, temperature_k=temperature)
    pmf_ab = compute_pmf_1d(samples_ab, temperature_k=temperature)
    result = cooperativity_free_energy(pmf_a, pmf_ab)

    click.echo(json.dumps(result.__dict__, indent=2, default=str))


@main.command()
@click.option("--trajectory", "trajectory_path", required=True, type=click.Path(exists=True))
@click.option("--topology", required=True, type=click.Path(exists=True))
@click.option("--selection", default="protein and name CA", show_default=True)
def allostery(trajectory_path: str, topology: str, selection: str) -> None:
    """Build a DCCM-based communication graph and report top-ranked pathways."""
    from enhancoai.md.loader import load_trajectory
    from enhancoai.md.correlations import dynamic_cross_correlation
    from enhancoai.allostery.network import build_communication_graph, graph_summary
    from enhancoai.allostery.pathways import betweenness_centrality

    handle = load_trajectory(topology, trajectory_path)
    dccm = dynamic_cross_correlation(handle, selection=selection)
    graph = build_communication_graph(correlation_matrix=dccm, correlation_chain_id="A")
    click.echo(json.dumps(graph_summary(graph), indent=2))

    centrality = betweenness_centrality(graph)
    top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    click.echo("\nTop betweenness-centrality nodes:")
    for node, score in top:
        click.echo(f"  {node}: {score:.4f}")


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--experiment-dir", default=None, type=click.Path())
def train(config_path: str, experiment_dir: str | None) -> None:
    """Train an EnhancoAI model from a training configuration file."""
    from enhancoai.utils.config import ProjectConfig
    from enhancoai.models.factory import build_model
    from enhancoai.training.trainer import Trainer
    from enhancoai.training.losses import LossWeights
    from enhancoai.utils.device import detect_device, describe

    config = ProjectConfig.from_yaml(config_path)
    device_info = detect_device()
    click.echo(describe(device_info))

    model = build_model(config.model)
    click.echo(f"Model: {config.model.architecture} ({sum(p.numel() for p in model.parameters()):,} parameters)")
    click.echo("Training requires a prepared dataset. See `python scripts/generate_demo_dataset.py` "
                "and `python scripts/train.py --config ...` for a runnable example.")


@main.command()
@click.option("--model", "model_path", required=True, type=click.Path())
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--architecture", default="hybrid", show_default=True)
def predict(model_path: str, input_path: str, architecture: str) -> None:
    """Run inference with a trained model on a prepared project directory."""
    click.echo(
        f"To predict, load features from '{input_path}' with the appropriate feature-extraction "
        f"module and call enhancoai.inference.predictor.Predictor('{model_path}', ...). "
        "See scripts/predict.py for a complete example."
    )


@main.command()
@click.option("--model", "model_path", required=True, type=click.Path())
@click.option("--target", default="cooperativity", show_default=True)
def explain(model_path: str, target: str) -> None:
    """Run explainability (integrated gradients / saliency / Grad-CAM) on a trained model."""
    click.echo(
        f"See scripts/generate_report.py or enhancoai.explainability for programmatic use. "
        f"Target task: {target}, model: {model_path}"
    )


@main.command()
@click.option("--project", "project_dir", required=True, type=click.Path(exists=True))
@click.option("--output", default=None, type=click.Path())
def report(project_dir: str, output: str | None) -> None:
    """Generate an HTML + PDF scientific report for a project."""
    click.echo(f"See scripts/generate_report.py for a full example against '{project_dir}'.")


if __name__ == "__main__":
    main()
