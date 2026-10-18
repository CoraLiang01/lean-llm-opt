File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 7
Columns: ['customer', 'demand']
Parsed column types: {'customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
customer demand
      C1   1083
      C2    776
      C3  16214
      C4    553
      C5  17106
      C6    594
      C7    732
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 7}, "demand": {"missing": 0, "unique_nonempty": 7, "numeric_range": [553.0, 17106.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 7
Columns: ['Unnamed: 0', 'fixed_costs']
Parsed column types: {'Unnamed: 0': 'object', 'fixed_costs': 'float64'}
Preview only (first 10 rows):
Unnamed: 0       fixed_costs
        S1            102.33
        S2             94.92
        S3             91.83
        S4 98.70999999999999
        S5             95.73
        S6 99.95999999999999
        S7             98.16
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 7}, "fixed_costs": {"missing": 0, "unique_nonempty": 7, "numeric_range": [91.83, 102.33]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 7
Columns: ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
Parsed column types: {'Unnamed: 0': 'object', 'C1': 'float64', 'C2': 'float64', 'C3': 'float64', 'C4': 'float64', 'C5': 'float64', 'C6': 'float64', 'C7': 'float64'}
Preview only (first 10 rows):
Unnamed: 0      C1                C2                C3                C4                C5                C6      C7
        S1 1506.22 70.90000000000001              8.44            260.27            197.47 71.70999999999999   61.19
        S2 1732.65           1780.72 567.4400000000001            448.68                29           1484.91  963.92
        S3  115.66            100.76 64.68000000000001           1324.53 64.98999999999999            134.88 2102.83
        S4 1254.78           1115.63             52.31           1036.16            892.63           1464.04 1383.41
        S5    42.9            891.01           1013.94           1128.72             58.91             42.89 1570.31
        S6     0.7            139.46             70.03 79.15000000000001              1482              0.91  110.46
        S7  1732.3           1780.44             486.5            523.74            522.08             82.48  826.41
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 7}, "C1": {"missing": 0, "unique_nonempty": 7, "numeric_range": [0.7, 1732.65]}, "C2": {"missing": 0, "unique_nonempty": 7, "numeric_range": [70.9, 1780.72]}, "C3": {"missing": 0, "unique_nonempty": 7, "numeric_range": [8.44, 1013.94]}, "C4": {"missing": 0, "unique_nonempty": 7, "numeric_range": [79.15, 1324.53]}, "C5": {"missing": 0, "unique_nonempty": 7, "numeric_range": [29.0, 1482.0]}, "C6": {"missing": 0, "unique_nonempty": 7, "numeric_range": [0.91, 1484.91]}, "C7": {"missing": 0, "unique_nonempty": 7, "numeric_range": [61.19, 2102.83]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]