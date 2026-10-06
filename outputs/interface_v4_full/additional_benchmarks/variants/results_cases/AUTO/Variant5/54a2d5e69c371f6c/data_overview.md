File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/machine_capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['Machine', 'Capacity']
Parsed column types: {'Machine': 'object', 'Capacity': 'int64'}
Preview only (first 10 rows):
Machine Capacity
     M1       13
     M2       12
     M3       12
     M4       12
Full-file column statistics: {"Machine": {"missing": 0, "unique_nonempty": 4}, "Capacity": {"missing": 0, "unique_nonempty": 2, "numeric_range": [12.0, 13.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/assignment_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['Machine', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8']
Parsed column types: {'Machine': 'object', 'J1': 'int64', 'J2': 'int64', 'J3': 'int64', 'J4': 'int64', 'J5': 'int64', 'J6': 'int64', 'J7': 'int64', 'J8': 'int64'}
Preview only (first 10 rows):
Machine J1 J2 J3 J4 J5 J6 J7 J8
     M1  8  7 25 24 27 26 28 29
     M2 23 24  6  9 25 27 26 28
     M3 27 26 24 25  5  8 23 24
     M4 25 27 26 24 23 25  6  7
Full-file column statistics: {"Machine": {"missing": 0, "unique_nonempty": 4}, "J1": {"missing": 0, "unique_nonempty": 4, "numeric_range": [8.0, 27.0]}, "J2": {"missing": 0, "unique_nonempty": 4, "numeric_range": [7.0, 27.0]}, "J3": {"missing": 0, "unique_nonempty": 4, "numeric_range": [6.0, 26.0]}, "J4": {"missing": 0, "unique_nonempty": 3, "numeric_range": [9.0, 25.0]}, "J5": {"missing": 0, "unique_nonempty": 4, "numeric_range": [5.0, 27.0]}, "J6": {"missing": 0, "unique_nonempty": 4, "numeric_range": [8.0, 27.0]}, "J7": {"missing": 0, "unique_nonempty": 4, "numeric_range": [6.0, 28.0]}, "J8": {"missing": 0, "unique_nonempty": 4, "numeric_range": [7.0, 29.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "j1", "matching_columns": [], "exact_matching_columns": 0}, {"term": "j8", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/assignment_resources.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['Machine', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8']
Parsed column types: {'Machine': 'object', 'J1': 'int64', 'J2': 'int64', 'J3': 'int64', 'J4': 'int64', 'J5': 'int64', 'J6': 'int64', 'J7': 'int64', 'J8': 'int64'}
Preview only (first 10 rows):
Machine J1 J2 J3 J4 J5 J6 J7 J8
     M1  5  6  8  7  9  8  7  7
     M2  8  7  4  7  8  9  8  7
     M3  9  8  7  8  6  5  8  7
     M4  8  8  7  8  8  7  5  6
Full-file column statistics: {"Machine": {"missing": 0, "unique_nonempty": 4}, "J1": {"missing": 0, "unique_nonempty": 3, "numeric_range": [5.0, 9.0]}, "J2": {"missing": 0, "unique_nonempty": 3, "numeric_range": [6.0, 8.0]}, "J3": {"missing": 0, "unique_nonempty": 3, "numeric_range": [4.0, 8.0]}, "J4": {"missing": 0, "unique_nonempty": 2, "numeric_range": [7.0, 8.0]}, "J5": {"missing": 0, "unique_nonempty": 3, "numeric_range": [6.0, 9.0]}, "J6": {"missing": 0, "unique_nonempty": 4, "numeric_range": [5.0, 9.0]}, "J7": {"missing": 0, "unique_nonempty": 3, "numeric_range": [5.0, 8.0]}, "J8": {"missing": 0, "unique_nonempty": 2, "numeric_range": [6.0, 7.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "j1", "matching_columns": [], "exact_matching_columns": 0}, {"term": "j8", "matching_columns": [], "exact_matching_columns": 0}]