File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant13/inputs/project_activities.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['Activity', 'Predecessors', 'NormalDuration', 'CrashDuration', 'CrashCostPerDay']
Parsed column types: {'Activity': 'object', 'Predecessors': 'object', 'NormalDuration': 'int64', 'CrashDuration': 'int64', 'CrashCostPerDay': 'int64'}
Preview only (first 10 rows):
Activity Predecessors NormalDuration CrashDuration CrashCostPerDay
       A                           5             3             180
       B                           6             4             150
       C            A              7             4             130
       D            A              4             3             210
       E            B              5             3             160
       F          C;D              6             4             190
       G          D;E              7             5             140
       H            F              4             2             220
       I            G              5             3             170
       J          H;I              3             2             260
Full-file column statistics: {"Activity": {"missing": 0, "unique_nonempty": 10}, "Predecessors": {"missing": 2, "unique_nonempty": 7}, "NormalDuration": {"missing": 0, "unique_nonempty": 5, "numeric_range": [3.0, 7.0]}, "CrashDuration": {"missing": 0, "unique_nonempty": 4, "numeric_range": [2.0, 5.0]}, "CrashCostPerDay": {"missing": 0, "unique_nonempty": 10, "numeric_range": [130.0, 260.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "a", "matching_columns": [{"column": "Activity", "exact": 1, "prefix": 1, "contains": 1, "examples": ["A"]}, {"column": "Predecessors", "exact": 2, "prefix": 2, "contains": 2, "examples": ["A"]}], "exact_matching_columns": 2}, {"term": "activity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "i", "matching_columns": [{"column": "Activity", "exact": 1, "prefix": 1, "contains": 1, "examples": ["I"]}, {"column": "Predecessors", "exact": 0, "prefix": 0, "contains": 1, "examples": ["H;I"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant13/inputs/project_parameters.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 1
Columns: ['Parameter', 'Value']
Parsed column types: {'Parameter': 'object', 'Value': 'int64'}
Preview only (first 10 rows):
      Parameter Value
ProjectDeadline    23
Full-file column statistics: {"Parameter": {"missing": 0, "unique_nonempty": 1}, "Value": {"missing": 0, "unique_nonempty": 1, "numeric_range": [23.0, 23.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: []