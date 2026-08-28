#!/usr/bin/env python
"""Generate a full EnhancoAI HTML + PDF report for a structure (section 48).

Example:
    python scripts/generate_report.py --input data/examples/demo_tf_dna_two_factor.pdb \\
        --protein A --protein B --dna C --dna D --output-dir experiments/report_demo
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--protein", action="append", required=True, dest="protein_chains")
    parser.add_argument("--dna", action="append", required=True, dest="dna_chains")
    parser.add_argument("--output-dir", default="experiments/report")
    args = parser.parse_args()

    from enhancoai.structure.parser import load_structure
    from enhancoai.structure.chain_detection import summarize_chains
    from enhancoai.structure.cleaning import report_missing_backbone_atoms
    from enhancoai.interactions.protein_dna import analyse_protein_dna_interactions, format_contact_descriptions
    from enhancoai.interactions.protein_protein import analyse_protein_protein_interactions
    from enhancoai.dna.geometry import find_watson_crick_pairs, base_pair_frames
    from enhancoai.dna.helical_parameters import step_parameters
    from enhancoai.visualization.structure import render_structure_matplotlib
    from enhancoai.visualization.contacts import render_contact_map
    from enhancoai.visualization.dna import plot_parameter_vs_position
    from enhancoai.reports import ReportData, ReportSection
    from enhancoai.reports.html import render_html_report
    from enhancoai.reports.pdf import render_pdf_report
    from enhancoai.utils.reproducibility import ReproducibilityRecord

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    structure = load_structure(args.input)
    sections = []

    fig = render_structure_matplotlib(structure, title="Full complex")
    fig.savefig(figures_dir / "structure.png", dpi=120)
    sections.append(
        ReportSection(
            title="Structure",
            html_body=summarize_chains(structure).to_html(index=False),
            figure_paths=[str(figures_dir / "structure.png")],
        )
    )

    missing = report_missing_backbone_atoms(structure)
    sections.append(
        ReportSection(
            title="TF/DNA Identification",
            html_body=f"<p>Protein chains: {', '.join(args.protein_chains)}<br>"
            f"DNA chains: {', '.join(args.dna_chains)}<br>"
            f"{len(missing)} residue(s) with missing backbone atoms.</p>",
        )
    )

    contact_html, contact_figures = [], []
    for protein_chain in args.protein_chains:
        for dna_chain in args.dna_chains:
            result = analyse_protein_dna_interactions(structure, protein_chain, dna_chain)
            fig = render_contact_map(result.residue_contact_map, title=f"{protein_chain} <-> {dna_chain}")
            fig_path = figures_dir / f"contacts_{protein_chain}_{dna_chain}.png"
            fig.savefig(fig_path, dpi=120)
            contact_html.append(f"<h3>{protein_chain} &lt;-&gt; {dna_chain}</h3><pre>{result.summary()}</pre>")
            contact_figures.append(str(fig_path))
    sections.append(ReportSection(title="Protein-DNA Contacts", html_body="".join(contact_html), figure_paths=contact_figures))

    pp_html = []
    for i, chain_a in enumerate(args.protein_chains):
        for chain_b in args.protein_chains[i + 1 :]:
            result = analyse_protein_protein_interactions(structure, chain_a, chain_b, compute_sasa=False)
            pp_html.append(f"<pre>{result.summary()}</pre>")
    sections.append(ReportSection(title="Protein-Protein Contacts", html_body="".join(pp_html) or "<p>Only one protein chain specified.</p>"))

    sections.append(ReportSection(title="MD Analysis", html_body="<p>No trajectory supplied to this report run; see `enhancoai md` / scripts/analyse_md.py.</p>"))

    dna_html = "<p>No paired DNA strands available for helical-parameter analysis.</p>"
    dna_figures = []
    if len(args.dna_chains) >= 2:
        pairs = find_watson_crick_pairs(structure, args.dna_chains[0], args.dna_chains[1])
        if pairs:
            frames = base_pair_frames(structure, args.dna_chains[0], args.dna_chains[1], pairs)
            steps = step_parameters(frames)
            fig = plot_parameter_vs_position(steps, "twist")
            fig_path = figures_dir / "dna_twist.png"
            fig.savefig(fig_path, dpi=120)
            dna_figures = [str(fig_path)]
            dna_html = f"<p>{len(pairs)} base pair(s) detected (simplified geometric analysis; see docs/dna_allostery.md).</p>"
    sections.append(ReportSection(title="DNA Dynamics", html_body=dna_html, figure_paths=dna_figures))

    sections.append(ReportSection(title="Orientation Analysis", html_body="<p>See `enhancoai.md.orientation` for TF-DNA / TF-TF orientation angle trajectories (requires an MD trajectory).</p>"))
    sections.append(ReportSection(title="Free-Energy Analysis", html_body="<p>No reaction-coordinate samples supplied to this report run; see `enhancoai cooperativity` / scripts/calculate_pmf.py.</p>"))
    sections.append(ReportSection(title="Cooperativity", html_body="<p>Requires comparing a TF-A-alone system against a TF-A+TF-B system (see docs/cooperativity.md); not computed in this single-structure report.</p>"))
    sections.append(ReportSection(title="Allosteric Network", html_body="<p>Requires an MD trajectory (DCCM/mutual-information edges); see `enhancoai allostery`.</p>"))
    sections.append(ReportSection(title="AI Predictions", html_body="<p>No trained model checkpoint supplied to this report run; see scripts/predict.py.</p>"))
    sections.append(ReportSection(title="Explainability", html_body="<p>Requires a trained model; see `enhancoai explain` / enhancoai.explainability.</p>"))
    sections.append(ReportSection(title="Important Residues", html_body="<p>Populated from explainability output when a trained model is supplied.</p>"))
    sections.append(ReportSection(title="Important DNA Bases", html_body="<p>Populated from explainability output when a trained model is supplied.</p>"))

    sections.append(
        ReportSection(
            title="Limitations",
            html_body=(
                "<ul>"
                "<li>Contact/hydrogen-bond/salt-bridge detection uses documented geometric criteria, not full electronic-structure chemistry.</li>"
                "<li>DNA helical parameters are simplified geometric approximations, not a 3DNA/Curves+ reimplementation.</li>"
                "<li>Free-energy and cooperativity estimates (when computed) are simulation-derived proxies, not experimental measurements.</li>"
                "<li>AI predictions depend entirely on training data and are meaningless without trained, validated weights.</li>"
                "<li>Correlation-based allosteric evidence does not by itself prove causal communication.</li>"
                "</ul>"
            ),
        )
    )

    record = ReproducibilityRecord()
    record.add_input("structure", args.input)
    record.parameters = {"protein_chains": args.protein_chains, "dna_chains": args.dna_chains}
    record.export(output_dir / "reproducibility.json")
    sections.append(
        ReportSection(
            title="Reproducibility Information",
            html_body=f"<pre>{record.to_dict()}</pre><p>Full record saved to reproducibility.json.</p>",
        )
    )

    data = ReportData(project_name="EnhancoAI", generated_at=datetime.now(timezone.utc).isoformat(), sections=sections)
    html_path = render_html_report(data, output_dir / "report.html")
    pdf_path = render_pdf_report(data, output_dir / "report.pdf")
    print(f"Wrote {html_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
