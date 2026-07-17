"""Module 9 : Orchestration du pipeline scientifique via Snakemake.

DAG : derive_symbolic -> ingest_and_vectorize -> analyze_stability
                      \-> train_pinn -> generate_plots

Exécution complète : `snakemake --cores all`
"""
configfile: "config.yaml"

rule all:
    input:
        "outputs/figures/solution_surface.pdf",
        "outputs/figures/visualisation_pinn_interactive.html",
        "outputs/stability_report.txt",

# Étape 1 : Dérivation symbolique du terme source (Module 3)
rule derive_symbolic:
    output:
        "outputs/reference.npz"
    shell:
        "python -m src.symbolic_derivation --output {output}"

# Étape 2 : Ingestion Polars + matrices de discrétisation (Module 4)
rule ingest_and_vectorize:
    input:
        sensors = "data/raw_sensors/sensors_sample.csv"
    output:
        "data/processed_grid.npy"
    shell:
        "python -m src.numerical_core --output {output} --sensors {input.sensors}"

# Étape 3 : Analyse de stabilité et conditionnement (Module 5)
rule analyze_stability:
    output:
        "outputs/stability_report.txt"
    shell:
        "python -m src.stability_analysis --output {output}"

# Étape 4 : Entraînement du PINN (Module 7)
rule train_pinn:
    input:
        grid = "data/processed_grid.npy",
        reference = "outputs/reference.npz"
    output:
        model = "models/pinn_weights.pth"
    shell:
        "python -m src.deep_pinn --save_path {output.model} --epochs 200"

# Étape 5 : Génération des figures PDF et HTML (Module 8)
rule generate_plots:
    input:
        weights = "models/pinn_weights.pth"
    output:
        pdf = "outputs/figures/solution_surface.pdf",
        html = "outputs/figures/visualisation_pinn_interactive.html"
    shell:
        "python -m src.visualisation --weights {input.weights} "
        "--output {output.pdf} --html {output.html}"
