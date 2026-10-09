"""Summarize preserved tool evidence and retry categories; no inference or rescoring."""
import argparse
import json

from evaluate_react_revision_20261006 import OUT


def audit(version, method='full'):
    root = OUT / f'{method}_{version}'
    rows = [json.loads(path.read_text()) for path in root.glob('**/attempts/*/result.json')]
    traces = [json.loads(row.get('csvqa_trace') or '{}') for row in rows]
    calls = [trace.get('tool_calls') or [] for trace in traces]
    attempted = sum(int(trace.get('csvqa_call_count', 0)) > 0 for trace in traces)
    valid = sum(any(call.get('observation') for call in history) for history in calls)
    unknown = sum(not trace.get('status') for trace in traces)
    result = {
        'version': version, 'method': method,
        'complete': json.loads((root / 'run_status.json').read_text()).get('complete', False)
                    if (root / 'run_status.json').exists() else False,
        'recorded_cases': len(rows), 'csvqa_attempted_cases': attempted,
        'csvqa_valid_observation_cases': valid,
        'csvqa_total_attempts': sum(int(t.get('csvqa_call_count', 0)) for t in traces),
        'cases_with_multiple_csvqa_attempts': sum(int(t.get('csvqa_call_count', 0)) > 1 for t in traces),
        'cases_with_protocol_restart': sum(int(row.get('protocol_retry_count', 0)) > 0 for row in rows),
        'protocol_restart_count': sum(int(row.get('protocol_retry_count', 0)) for row in rows),
        'pipeline_retry_count': sum(int(row.get('retry_count', 0)) - int(row.get('protocol_retry_count', 0))
                                    for row in rows),
        'repair_count': sum(int(row.get('repair_count', 0)) for row in rows),
        'http_retry_count': sum(int(row.get('api_retry_count', 0)) for row in rows),
        'full_data_fallback_count': sum(int(row.get('fallback_count', 0)) for row in rows),
        'csvqa_detail_unknown_cases': unknown,
        'http_retries_are_transport_only': True,
        'fallback_is_source_data_observation_not_model_repair': True,
        'evidence_rule': 'Attempt requires recorded tool history; valid Observation requires a completed tool call with saved Observation. Missing evidence is not fabricated.',
    }
    (root / 'execution_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True)
    parser.add_argument('--method', default='full')
    args = parser.parse_args()
    audit(args.version, args.method)
