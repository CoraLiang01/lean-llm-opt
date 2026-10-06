File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant3/inputs/project_activities.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 12
Columns: ['Activity', 'Predecessors', 'NormalDuration', 'CrashDuration', 'CrashCostPerDay']
Parsed column types: {'Activity': 'object', 'Predecessors': 'object', 'NormalDuration': 'int64', 'CrashDuration': 'int64', 'CrashCostPerDay': 'int64'}
Preview only (first 10 rows):
Activity Predecessors NormalDuration CrashDuration CrashCostPerDay
       A                           6             4             300
       B                           5             4             250
       C            A              7             4             180
       D            A              4             3             220
       E            B              6             4             160
       F            B              8             5             140
       G          C;D              5             3             210
       H          D;E              7             4             190
       I            F              6             4             170
       J          G;H              4             3             260
Full-file column statistics: {"Activity": {"missing": 0, "unique_nonempty": 12}, "Predecessors": {"missing": 2, "unique_nonempty": 8}, "NormalDuration": {"missing": 0, "unique_nonempty": 6, "numeric_range": [3.0, 8.0]}, "CrashDuration": {"missing": 0, "unique_nonempty": 4, "numeric_range": [2.0, 5.0]}, "CrashCostPerDay": {"missing": 0, "unique_nonempty": 12, "numeric_range": [140.0, 320.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "a", "matching_columns": [{"column": "Activity", "exact": 1, "prefix": 1, "contains": 1, "examples": ["A"]}, {"column": "Predecessors", "exact": 2, "prefix": 2, "contains": 2, "examples": ["A"]}], "exact_matching_columns": 2}, {"term": "activity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "i", "matching_columns": [{"column": "Activity", "exact": 1, "prefix": 1, "contains": 1, "examples": ["I"]}, {"column": "Predecessors", "exact": 0, "prefix": 0, "contains": 1, "examples": ["H;I"]}], "exact_matching_columns": 1}, {"term": "predecessors", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant3/inputs/project_parameters.csv
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