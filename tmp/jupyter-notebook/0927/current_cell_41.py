def loto_preflight():
    """Validate the seven disjoint gold-label folds and CSV paths without model calls."""
    frame = load_benchmark()
    if len(frame) != 101 or frame["problem_id"].duplicated().any():
        raise ValueError("Expected 101 cases with unique problem IDs")
    if frame["true_label"].isna().any() or not frame["true_label"].isin(CLASS_LABELS).all():
        raise ValueError("Every LOTO case needs a supported gold semantic label for grouping")
    for address in frame["dataset_address"]:
        paths = normalize_data_address(address).splitlines()
        if not paths:
            raise ValueError("These notebooks evaluate only the 101 cases with CSV inputs")
        for raw in paths:
            if not Path(raw).is_file():
                raise FileNotFoundError(raw)
    rows = []
    for held_out in CLASS_LABELS:
        manifest = set_loto_fold(held_out)
        rows.append({"held_out_type": held_out, "cases": int(frame["true_label"].eq(held_out).sum()),
                     "examples_removed": len(manifest["removed_examples"]),
                     "examples_remaining": manifest["reference_count_after"],
                     "disabled_route": manifest["disabled_workflow_route"],
                     "allowed_labels": ",".join(manifest["allowed_labels"])})
    preview = pd.DataFrame(rows)
    if preview["cases"].sum() != 101 or preview["cases"].eq(0).any():
        raise ValueError("LOTO folds must partition all 101 cases without empty types")
    return frame, preview


def loto_report(records, output_dir):
    frame = pd.DataFrame(records)
    if "pipeline_stage" not in frame:
        frame["pipeline_stage"] = "setup"
    else:
        frame["pipeline_stage"] = frame["pipeline_stage"].fillna("setup")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for held_out, group in frame.groupby("held_out_type", sort=False):
        total = len(group)
        rows.append({"held_out_type": held_out, "cases": total,
                     "solved": int(group["final_ok"].eq(True).sum()),
                     "objective_correct": int(group["solution_correct"].eq(True).sum()),
                     "solve_rate": float(group["final_ok"].eq(True).mean()),
                     "objective_accuracy": float(group["solution_correct"].eq(True).mean())})
    per_type = pd.DataFrame(rows)
    summary_rows = []
    for label, field in [("Solved", "final_ok"), ("Objective match", "solution_correct")]:
        count = int(frame[field].eq(True).sum())
        summary_rows.append({"metric": label, "correct": count, "total": len(frame),
                             "accuracy": count / len(frame)})
    summary_rows.append({"metric": "Classification", "correct": int(frame["classification_correct"].eq(True).sum()),
                         "total": len(frame), "accuracy": float(frame["classification_correct"].eq(True).mean())})
    summary_rows.append({"metric": "Macro objective match across evaluated types", "correct": None,
                         "total": len(per_type), "accuracy": float(per_type["objective_accuracy"].mean())})
    summary = pd.DataFrame(summary_rows)
    routes = frame.groupby(["held_out_type", "assigned_route"], dropna=False).size().rename("cases").reset_index()
    failures = frame.loc[~frame["final_ok"].eq(True)].groupby(
        ["held_out_type", "pipeline_stage", "execution_error_type"], dropna=False
    ).size().rename("cases").reset_index() if "execution_error_type" in frame else pd.DataFrame()
    columns = [column for column in (
        "problem_id", "held_out_type", "loto_variant", "true_label", "predicted_label",
        "assigned_route", "disabled_workflow_route", "allowed_labels", "route_allowed",
        "final_ok", "final_objective", "label_objective", "solution_correct", "classification_correct",
        "pipeline_stage", "execution_error_type", "execution_error", "cache_source", "cache_fingerprint",
    ) if column in frame]
    tables = {"summary": summary, "accuracy_by_held_out_type": per_type,
              "selected_routes": routes, "failures": failures, "case_summary": frame[columns]}
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    display(per_type)
    display(summary)
    return tables


def run_loto(test=None, *, output_dir=None, folds=None, rows=None, reuse_completed=True, continue_on_error=True,
             rel_tol=1e-4, abs_tol=1e-4):
    """Fresh classification per case/fold; resume only matching, complete successful results."""
    frame = load_benchmark() if test is None else prepare_cases(test)
    validate_loto_cases(frame)
    destination = RESULTS_DIR if output_dir is None else Path(output_dir)
    chosen = list(CLASS_LABELS) if folds is None else [normalize_problem_class(value) for value in folds]
    if not chosen or len(chosen) != len(set(chosen)) or any(value not in CLASS_LABELS for value in chosen):
        raise ValueError("FOLDS must contain distinct supported semantic types")
    if rows is not None:
        rows = list(rows)
        if (len(rows) != len(set(rows)) or any(isinstance(row, (bool, np.bool_))
                or not isinstance(row, (int, np.integer)) or not 0 <= row < len(frame) for row in rows)):
            raise ValueError("ROWS must be distinct valid zero-based row indices")
        frame = frame.iloc[rows]
    selected = frame.loc[frame["true_label"].isin(chosen)]
    if selected.empty:
        raise ValueError("FOLDS and ROWS select no cases")
    source = _source_fingerprint()
    results, api_checked = [], False
    for held_out in chosen:
        fold_cases = selected.loc[selected["true_label"].eq(held_out)]
        if fold_cases.empty:
            continue
        manifest = set_loto_fold(held_out)
        folder = destination / held_out
        folder.mkdir(parents=True, exist_ok=True)
        manifest.update(case_ids=fold_cases["problem_id"].tolist(), source_fingerprint=source,
                        rel_tol=rel_tol, abs_tol=abs_tol)
        _write_text_atomic(folder / "fold_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
        output_csv = folder / "results.csv"
        existing = load_records(output_csv)
        if existing and not reuse_completed:
            raise ValueError("Use a fresh round directory; recorded LOTO attempts cannot be overwritten")
        cached = {record_key(record): record for record in existing}
        for case in fold_cases.to_dict("records"):
            fingerprint = case_fingerprint(case, source_fingerprint=source)
            key = (case["problem_id"], str(case["Query"]), case["dataset_address"], "AUTO")
            previous = cached.get(key)
            if previous and previous.get("cache_fingerprint") != fingerprint:
                raise ValueError("LOTO code/data changed; use a fresh version directory")
            record = None
            if (previous and previous.get("record_status") in {"completed", "error"}
                    and previous.get("cache_fingerprint") == fingerprint):
                record = materialize_record(output_csv, previous)
                if record is not None:
                    record.update(cache_source="csv", **loto_record_fields())
            if record is None:
                if not api_checked and callable(globals().get("require_api_key")):
                    require_api_key()
                api_checked = True
                try:
                    record = execute_pipeline_case(case)
                except Exception as exc:
                    record = error_record(case, exc)
                    record.update(loto_record_fields())
                    if not continue_on_error:
                        record = finalize_record(record, rel_tol=rel_tol, abs_tol=abs_tol)
                        record["cache_fingerprint"] = fingerprint
                        save_record(output_csv, record)
                        raise
            record = finalize_record(record, rel_tol=rel_tol, abs_tol=abs_tol)
            record["cache_fingerprint"] = fingerprint
            save_record(output_csv, record)
            results.append(record)
            print(f"[{len(results)}/{len(selected)}] {held_out} / {case['problem_id']}: "
                  f"{record['record_status']} via {record.get('assigned_route')} [{record.get('cache_source')}]")
    loto_report(results, destination)
    return results


def run_test(*args, **kwargs):
    raise RuntimeError("Use run_loto(folds=..., rows=...) to preserve fold and cache isolation")


def validate_loto_cases(frame):
    if frame.empty or frame["problem_id"].duplicated().any():
        raise ValueError("LOTO requires nonempty unique case IDs")
    if frame["true_label"].isna().any() or not frame["true_label"].isin(CLASS_LABELS).all():
        raise ValueError("Every evaluation case needs a declared semantic type for fold grouping")
    for address in frame["dataset_address"]:
        if not address or any(not Path(raw).is_file() for raw in address.splitlines()):
            raise FileNotFoundError(f"Missing current-case CSV data: {address}")


_BASE_PREPARE_CASES = prepare_cases

def prepare_cases(test, *, dataset_root=None):
    if dataset_root is None:
        return _BASE_PREPARE_CASES(test)
    frame = test.copy()
    address_column = "dataset_address" if "dataset_address" in frame else "Dataset_address"
    def resolve_relocated(value):
        paths = []
        root = Path(dataset_root).resolve()
        for raw in normalize_data_address(value).splitlines():
            path = Path(raw)
            standard = path if path.is_absolute() else PROJECT_ROOT / path
            if standard.is_file():
                paths.append(str(standard.resolve())); continue
            candidates = [root/path, root.joinpath(*path.parts[1:])]
            matches = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
            if len(matches) != 1:
                raise ValueError(f"Cannot uniquely resolve relocated data: {raw}")
            paths.append(str(matches[0]))
        return "\n".join(paths)
    frame[address_column] = frame[address_column].map(resolve_relocated)
    return _BASE_PREPARE_CASES(frame)
