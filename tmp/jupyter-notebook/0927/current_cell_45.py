# To run only this benchmark, execute the preceding definition cells, then this cell.
RUN_VARIANTS = False
VARIANTS_FILE = PROJECT_ROOT / "benchmark_dataset/questions.csv"
VARIANTS_OUTPUT_DIR = RESULTS_DIR / "additional_benchmarks/variants"


def load_variants_for_baseline(path):
    frame = load_benchmark(path)
    problem_ids = []
    for address in frame["dataset_address"]:
        paths = [Path(line) for line in normalize_data_address(address).splitlines()]
        if not paths or any(not source.is_file() for source in paths):
            raise FileNotFoundError(f"Missing variants data files: {address}")
        ids = {part for source in paths for part in source.parts
               if re.fullmatch(r"Variant[0-9]+", part)}
        if len(ids) != 1:
            raise ValueError(f"Cannot determine a unique Variant ID: {address}")
        problem_ids.append(next(iter(ids)))
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError("Duplicate Variant IDs; check questions.csv")
    frame["problem_id"] = problem_ids
    return frame


if RUN_VARIANTS:
    variants_cases = load_variants_for_baseline(VARIANTS_FILE)
    print(f"Variants: {len(variants_cases)} cases; output directory: {VARIANTS_OUTPUT_DIR}")
    variants_results = run_loto(
        variants_cases, output_dir=VARIANTS_OUTPUT_DIR,
    )
    loto_report(variants_results, VARIANTS_OUTPUT_DIR)
