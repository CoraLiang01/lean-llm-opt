RUN_REDUNDANT_COLUMNS = False
COLUMNS_FILE = PROJECT_ROOT / "redundancy_complete/redundant_instances.xlsx"
COLUMN_SHEETS = [f"{pct}-S{seed}" for pct in ["50pct", "100pct", "200pct"] for seed in [1, 2, 3]]  # Select one or more sheets.
# Last three sheets: COLUMN_SHEETS = ["200pct-S1", "200pct-S2", "200pct-S3"]
# All nine sheets: COLUMN_SHEETS = [f"{pct}-S{seed}" for pct in ["50pct", "100pct", "200pct"] for seed in [1, 2, 3]]
COLUMNS_OUTPUT_DIR = RESULTS_DIR / "additional_benchmarks/columns"


def load_redundant_sheets_for_baseline(path, sheet_names):
    path = Path(path).expanduser()
    path = path if path.is_absolute() else PROJECT_ROOT / path
    selected = [sheet_names] if isinstance(sheet_names, str) else list(sheet_names)
    if not selected or any(not isinstance(name, str) or not name.strip() for name in selected):
        raise ValueError("COLUMN_SHEETS must contain at least one non-empty sheet name")
    if len(selected) != len(set(selected)):
        raise ValueError("COLUMN_SHEETS must not contain duplicate sheet names")
    frames = {}
    with pd.ExcelFile(path) as workbook:
        missing = [name for name in selected if name not in workbook.sheet_names]
        if missing:
            raise ValueError(f"Unknown sheets: {missing}; available sheets: {workbook.sheet_names}")
        for name in selected:
            # dataset_root resolves legacy directory names under the current workbook directory.
            frame = prepare_cases(pd.read_excel(workbook, sheet_name=name), dataset_root=path.parent)
            frame["benchmark_sheet"] = name
            frame["problem_id"] = name + "/" + frame["problem_id"]
            for address in frame["dataset_address"]:
                sources = [Path(line) for line in normalize_data_address(address).splitlines()]
                if not sources or any(not source.is_file() for source in sources):
                    raise FileNotFoundError(f"Missing data files for {name}: {address}")
            frames[name] = frame
    return frames


if RUN_REDUNDANT_COLUMNS:
    # Validate every selected sheet and data file before making model requests.
    column_cases_by_sheet = load_redundant_sheets_for_baseline(COLUMNS_FILE, COLUMN_SHEETS)
    redundant_results_by_sheet = {}
    for sheet_name, column_cases in column_cases_by_sheet.items():
        sheet_output_dir = COLUMNS_OUTPUT_DIR / sheet_name
        print(f"Redundant columns {sheet_name}: {len(column_cases)} cases; output directory: {sheet_output_dir}")
        sheet_results = run_loto(column_cases, output_dir=sheet_output_dir)
        redundant_results_by_sheet[sheet_name] = sheet_results
        loto_report(sheet_results, sheet_output_dir)
