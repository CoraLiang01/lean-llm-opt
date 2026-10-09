File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 3
Columns: ['customer', 'demand']
Parsed column types: {'customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
customer demand
      C1   1083
      C2    776
      C3  16214
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 3}, "demand": {"missing": 0, "unique_nonempty": 3, "numeric_range": [776.0, 16214.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 3
Columns: ['Unnamed: 0', 'fixed_costs']
Parsed column types: {'Unnamed: 0': 'object', 'fixed_costs': 'float64'}
Preview only (first 10 rows):
Unnamed: 0 fixed_costs
        S1      102.33
        S2       94.92
        S3       91.83
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 3}, "fixed_costs": {"missing": 0, "unique_nonempty": 3, "numeric_range": [91.83, 102.33]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 3
Columns: ['Unnamed: 0', 'C1', 'C2', 'C3']
Parsed column types: {'Unnamed: 0': 'object', 'C1': 'float64', 'C2': 'float64', 'C3': 'float64'}
Preview only (first 10 rows):
Unnamed: 0      C1                C2                C3
        S1 1506.22 70.90000000000001              8.44
        S2 1732.65           1780.72 567.4400000000001
        S3  115.66            100.76 64.68000000000001
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 3}, "C1": {"missing": 0, "unique_nonempty": 3, "numeric_range": [115.66, 1732.65]}, "C2": {"missing": 0, "unique_nonempty": 3, "numeric_range": [70.9, 1780.72]}, "C3": {"missing": 0, "unique_nonempty": 3, "numeric_range": [8.44, 567.44]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]