File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant15/inputs/workstation_capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['Workstation', 'Capacity']
Parsed column types: {'Workstation': 'object', 'Capacity': 'int64'}
Preview only (first 10 rows):
Workstation Capacity
         W1       15
         W2       14
         W3       16
         W4       13
Full-file column statistics: {"Workstation": {"missing": 0, "unique_nonempty": 4}, "Capacity": {"missing": 0, "unique_nonempty": 4, "numeric_range": [13.0, 16.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "workstation", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant15/inputs/assignment_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['Workstation', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9']
Parsed column types: {'Workstation': 'object', 'J1': 'int64', 'J2': 'int64', 'J3': 'int64', 'J4': 'int64', 'J5': 'int64', 'J6': 'int64', 'J7': 'int64', 'J8': 'int64', 'J9': 'int64'}
Preview only (first 10 rows):
Workstation J1 J2 J3 J4 J5 J6 J7 J8 J9
         W1  6  8 18 20 21 19 23 22 24
         W2 19 18  7  6 20 22 21 23 25
         W3 22 21 20 19  5  7 18 20 21
         W4 21 22 23 20 19 18  6  8  7
Full-file column statistics: {"Workstation": {"missing": 0, "unique_nonempty": 4}, "J1": {"missing": 0, "unique_nonempty": 4, "numeric_range": [6.0, 22.0]}, "J2": {"missing": 0, "unique_nonempty": 4, "numeric_range": [8.0, 22.0]}, "J3": {"missing": 0, "unique_nonempty": 4, "numeric_range": [7.0, 23.0]}, "J4": {"missing": 0, "unique_nonempty": 3, "numeric_range": [6.0, 20.0]}, "J5": {"missing": 0, "unique_nonempty": 4, "numeric_range": [5.0, 21.0]}, "J6": {"missing": 0, "unique_nonempty": 4, "numeric_range": [7.0, 22.0]}, "J7": {"missing": 0, "unique_nonempty": 4, "numeric_range": [6.0, 23.0]}, "J8": {"missing": 0, "unique_nonempty": 4, "numeric_range": [8.0, 23.0]}, "J9": {"missing": 0, "unique_nonempty": 4, "numeric_range": [7.0, 25.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "workstation", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant15/inputs/assignment_resources.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['Workstation', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9']
Parsed column types: {'Workstation': 'object', 'J1': 'int64', 'J2': 'int64', 'J3': 'int64', 'J4': 'int64', 'J5': 'int64', 'J6': 'int64', 'J7': 'int64', 'J8': 'int64', 'J9': 'int64'}
Preview only (first 10 rows):
Workstation J1 J2 J3 J4 J5 J6 J7 J8 J9
         W1  4  5  7  8  8  7  8  9  8
         W2  7  8  4  5  8  8  7  8  9
         W3  8  7  8  7  5  4  7  8  7
         W4  8  8  9  8  7  7  4  5  4
Full-file column statistics: {"Workstation": {"missing": 0, "unique_nonempty": 4}, "J1": {"missing": 0, "unique_nonempty": 3, "numeric_range": [4.0, 8.0]}, "J2": {"missing": 0, "unique_nonempty": 3, "numeric_range": [5.0, 8.0]}, "J3": {"missing": 0, "unique_nonempty": 4, "numeric_range": [4.0, 9.0]}, "J4": {"missing": 0, "unique_nonempty": 3, "numeric_range": [5.0, 8.0]}, "J5": {"missing": 0, "unique_nonempty": 3, "numeric_range": [5.0, 8.0]}, "J6": {"missing": 0, "unique_nonempty": 3, "numeric_range": [4.0, 8.0]}, "J7": {"missing": 0, "unique_nonempty": 3, "numeric_range": [4.0, 8.0]}, "J8": {"missing": 0, "unique_nonempty": 3, "numeric_range": [5.0, 9.0]}, "J9": {"missing": 0, "unique_nonempty": 4, "numeric_range": [4.0, 9.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "workstation", "matching_columns": [], "exact_matching_columns": 0}]