File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5
Columns: ['customer', 'demand']
Parsed column types: {'customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
customer demand
      C1    143
      C2      6
      C3     10
      C4     25
      C5      3
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 5}, "demand": {"missing": 0, "unique_nonempty": 5, "numeric_range": [3.0, 143.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5
Columns: ['Unnamed: 0', 'fixed_costs']
Parsed column types: {'Unnamed: 0': 'object', 'fixed_costs': 'float64'}
Preview only (first 10 rows):
Unnamed: 0       fixed_costs
        S1 97.65000000000001
        S2 99.76000000000001
        S3            100.76
        S4            105.32
        S5             98.88
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 5}, "fixed_costs": {"missing": 0, "unique_nonempty": 5, "numeric_range": [97.65, 105.32]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5
Columns: ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5']
Parsed column types: {'Unnamed: 0': 'object', 'C1': 'float64', 'C2': 'float64', 'C3': 'float64', 'C4': 'float64', 'C5': 'float64'}
Preview only (first 10 rows):
Unnamed: 0      C1                C2     C3      C4                C5
        S1  150.74              0.02  49.13 2080.15             426.4
        S2  233.05             97.73  49.84 1982.39             23.96
        S3   55.68            935.61   4.03   73.09 525.3200000000001
        S4 1483.82           1801.08 112.16  816.05            107.01
        S5 1119.47 884.3099999999999   0.08 1544.95            543.67
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 5}, "C1": {"missing": 0, "unique_nonempty": 5, "numeric_range": [55.68, 1483.82]}, "C2": {"missing": 0, "unique_nonempty": 5, "numeric_range": [0.02, 1801.08]}, "C3": {"missing": 0, "unique_nonempty": 5, "numeric_range": [0.08, 112.16]}, "C4": {"missing": 0, "unique_nonempty": 5, "numeric_range": [73.09, 2080.15]}, "C5": {"missing": 0, "unique_nonempty": 5, "numeric_range": [23.96, 543.67]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]