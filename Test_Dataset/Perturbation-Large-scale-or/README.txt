Perturbation Experiment 2: 303-Instance Public Release
======================================================

1. Package Contents
-------------------

This directory contains the complete public release of Perturbation Experiment 2.

1. test_set_303.csv
   The Gold-free test table. It contains 303 rows, with one perturbed problem
   instance per row.

2. datasets/
   The input data files used by the 303 test instances. The directory contains
   303 variant subdirectories and 546 CSV files. The Dataset_address field in
   test_set_303.csv lists the CSV file or files required by each instance.
   Multiple paths in one cell are separated by line breaks. All paths are
   relative to this directory.

3. gold_303.csv
   The reference answers for all 303 instances. It joins one-to-one with
   test_set_303.csv through variant_id. Label-objective is the reference optimal
   objective value, and Label-model is the reference Gurobi LP model.

Do not expose gold_303.csv, Label-objective, or Label-model to the evaluated
model during inference. Gold must only be opened after all inference outputs
have been completed and frozen.


2. Seeds and Instance IDs
-------------------------

- The source benchmark contains 101 original problems.
- The data seeds are 17, 42, and 73.
- Each seed creates one independent and reproducible perturbation of every one
  of the 101 source problems. Therefore, the release contains 101 x 3 = 303
  test instances.
- Each seed contributes exactly 101 rows.
- The three seeds are repeated perturbation draws. They are not difficulty
  levels and are not train, validation, and test splits.
- data_seed controls data generation only. It is separate from any random seed
  used by the evaluated language model.
- Variant IDs follow this format:

      additive__<instance_id>__seed<seed>

  Example:

      additive__TP1__seed17


3. Perturbation Rule
--------------------

All semantically mutable primitive business parameters in both the objective
and the constraints are perturbed jointly:

    new_value = original_value + d

where:

    d is one of {-2, -1, +1, +2}

Detailed rules:

- The offset for each parameter is selected deterministically from the data
  seed, instance_id, parameter_key, and resampling attempt using SHA256.
- Mutable quantities include prices, costs, profits, demand, supply, capacity,
  bounds, resource consumption, processing time, and business ratios.
- Structural constants, variable domains, constraint forms, structural zeros,
  and definitional constants are not perturbed.
- A business quantity represented in both the Query and a CSV file receives the
  same offset in both locations.
- Integer quantities remain integers. Decimal quantities retain their required
  precision.
- Invalid changes, such as negative demand, negative capacity, or out-of-range
  percentages, are excluded before an offset is selected.
- Derived values are recomputed from the perturbed primitive values rather than
  receiving independent offsets.
- Mixture17 uses a workday-conserving transfer: every worker column continues
  to sum to 10, while each directly changed cell changes by only 1 or 2.
- A candidate is deterministically resampled if it is infeasible, lacks a
  finite optimum, or duplicates another seed variant of the same source case.

This package is derived from the revised release:

    perturbation_exp2_additive_full101_schemafix_queryclarity_r2

All 303 Gold models were solved and verified as finite OPTIMAL solutions with
Gurobi before release.


4. Running the Public Main Experiment Notebook
----------------------------------------------

These instructions match the notebook currently available at the repository
root:

    LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb

Start Jupyter from the repository root. Run the notebook setup and function
definition cells through the cell that defines run_test and run_gurobi_code.
Do not run the original cell that loads:

    Test_Dataset/Large-scale-or/Large-scale-or-101.csv

Instead, add a new cell and run the following code. It resolves all dataset
paths correctly and processes the release separately by data seed.

    from pathlib import Path
    import pandas as pd

    PROJECT_ROOT = Path.cwd().resolve()
    DATASET_ROOT = (
        PROJECT_ROOT / "Test_Dataset/Perturbation-Large-scale-or"
    ).resolve()

    all_test = pd.read_csv(
        DATASET_ROOT / "test_set_303.csv",
        encoding="utf-8-sig",
    )

    assert len(all_test) == 303
    assert all_test["variant_id"].is_unique
    assert all_test.groupby("data_seed").size().to_dict() == {
        17: 101,
        42: 101,
        73: 101,
    }

    def make_seed_test(seed):
        frame = all_test.loc[all_test["data_seed"] == seed].copy()
        frame["Dataset_address"] = frame["Dataset_address"].map(
            lambda value: "\n".join(
                str((DATASET_ROOT / Path(line.strip())).resolve())
                for line in str(value).splitlines()
                if line.strip()
            )
        )
        return frame

    def execute_generated_code(code):
        if not isinstance(code, str) or not code.strip():
            return None
        extracted = extract_python_code(code)
        return run_gurobi_code(extracted if extracted is not None else code)

    for seed in (17, 42, 73):
        seed_test = make_seed_test(seed)

        output_model, output_code, classification = run_test(
            seed_test,
            classify_problem,
        )

        best_objective = [
            execute_generated_code(code)
            for code in output_code
        ]

        seed_results = pd.DataFrame({
            "variant_id": seed_test["variant_id"].to_numpy(),
            "data_seed": seed_test["data_seed"].to_numpy(),
            "Query": seed_test["Query"].to_numpy(),
            "model_output": output_model,
            "code_output": output_code,
            "classification": classification,
            "best_objective": best_objective,
        })

        assert len(seed_results) == 101
        seed_results.to_csv(
            LOG.dir / f"Perturbation-Large-scale-or-seed{seed}.csv",
            index=False,
            encoding="utf-8-sig",
        )

The expected inference files are:

    Perturbation-Large-scale-or-seed17.csv
    Perturbation-Large-scale-or-seed42.csv
    Perturbation-Large-scale-or-seed73.csv

Each file must contain exactly 101 rows. During inference, use only
test_set_303.csv and datasets/. Do not load gold_303.csv yet.


5. Scoring After Inference
--------------------------

Only after all three inference files are complete and frozen, run the following
cell to join the results with Gold and calculate objective accuracy:

    import numpy as np
    import pandas as pd

    result_parts = [
        pd.read_csv(
            LOG.dir / f"Perturbation-Large-scale-or-seed{seed}.csv",
            encoding="utf-8-sig",
        )
        for seed in (17, 42, 73)
    ]

    results_303 = pd.concat(result_parts, ignore_index=True)

    assert len(results_303) == 303
    assert results_303["variant_id"].is_unique

    gold_303 = pd.read_csv(
        DATASET_ROOT / "gold_303.csv",
        usecols=[
            "variant_id",
            "gold_status",
            "objective_sense",
            "Label-objective",
        ],
        encoding="utf-8-sig",
    )

    assert len(gold_303) == 303
    assert gold_303["variant_id"].is_unique
    assert gold_303["gold_status"].eq("OPTIMAL").all()

    scored_303 = results_303.merge(
        gold_303,
        on="variant_id",
        how="left",
        validate="one_to_one",
    )

    actual = pd.to_numeric(
        scored_303["best_objective"],
        errors="coerce",
    )
    expected = pd.to_numeric(
        scored_303["Label-objective"],
        errors="coerce",
    )

    scored_303["objective_correct"] = np.isclose(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
        equal_nan=False,
    )

    correct = int(scored_303["objective_correct"].sum())
    accuracy = scored_303["objective_correct"].mean()
    print(f"Correct: {correct}/303; accuracy: {accuracy:.2%}")

    scored_303.to_csv(
        LOG.dir / "Perturbation-Large-scale-or-scored-303.csv",
        index=False,
        encoding="utf-8-sig",
    )


6. Evaluation Integrity Requirements
------------------------------------

1. Do not merge test_set_303.csv with gold_303.csv before inference.
2. Do not place Label-objective or Label-model in the model prompt, retrieval
   corpus, fallback mechanism, answer-selection logic, or repair logic.
3. Keep data seed fixed by variant_id. Do not regenerate or substitute failed
   instances during evaluation.
4. When comparing systems, use the same 303 variant_ids, the same order, and
   the same numerical scoring tolerance.
5. Count failed, missing, non-optimal, and non-numeric outputs as incorrect;
   do not silently remove them from the denominator.
