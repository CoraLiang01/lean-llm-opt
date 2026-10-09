File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 9
Columns: ['customer', 'demand']
Parsed column types: {'customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
customer      demand
      C1  4742532000
      C2  1600594000
      C3  5086889000
      C4  1027326000
      C5 11926044000
      C6  9058407000
      C7  5344367000
      C8   677201000
      C9  3236493000
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 9}, "demand": {"missing": 0, "unique_nonempty": 9, "numeric_range": [677201000.0, 11926044000.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['Unnamed: 0', 'fixed_costs']
Parsed column types: {'Unnamed: 0': 'object', 'fixed_costs': 'float64'}
Preview only (first 10 rows):
Unnamed: 0 fixed_costs
        S1      100.64
        S2       98.72
        S3      100.18
        S4       96.58
        S5       95.75
        S6       99.06
        S7      101.78
        S8       93.86
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 8}, "fixed_costs": {"missing": 0, "unique_nonempty": 8, "numeric_range": [93.86, 101.78]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
Parsed column types: {'Unnamed: 0': 'object', 'C1': 'float64', 'C2': 'float64', 'C3': 'float64', 'C4': 'float64', 'C5': 'float64', 'C6': 'float64', 'C7': 'float64', 'C8': 'float64', 'C9': 'float64'}
Preview only (first 10 rows):
Unnamed: 0      C1      C2      C3      C4      C5      C6      C7      C8      C9
        S1 1091.04   85.72   99.08  747.35  893.86   23.65   15.11   15.03  497.88
        S2   58.88 1617.16 1786.44  951.81   56.45  642.77   16.69    0.63    11.2
        S3  110.47    0.04   38.89 1397.95 2361.45  107.62  1598.5   76.41 1382.84
        S4 1458.85 1049.27  597.32  1731.9   69.09 1227.17 1187.55 1017.16   52.15
        S5    0.38 2315.52 1313.06 1253.71   50.24   29.19   60.17 1077.35   70.11
        S6    58.2 1395.81    84.6  830.64 1003.86  631.17   31.13     1.4  246.24
        S7 1255.23 1382.31   78.79  829.02   67.31  877.35  185.28  221.98    0.05
        S8 1990.09    1.23   38.97 1396.35  112.54  107.54 1596.74   76.32 1183.79
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 8}, "C1": {"missing": 0, "unique_nonempty": 8, "numeric_range": [0.38, 1990.09]}, "C2": {"missing": 0, "unique_nonempty": 8, "numeric_range": [0.04, 2315.52]}, "C3": {"missing": 0, "unique_nonempty": 8, "numeric_range": [38.89, 1786.44]}, "C4": {"missing": 0, "unique_nonempty": 8, "numeric_range": [747.35, 1731.9]}, "C5": {"missing": 0, "unique_nonempty": 8, "numeric_range": [50.24, 2361.45]}, "C6": {"missing": 0, "unique_nonempty": 8, "numeric_range": [23.65, 1227.17]}, "C7": {"missing": 0, "unique_nonempty": 8, "numeric_range": [15.11, 1598.5]}, "C8": {"missing": 0, "unique_nonempty": 8, "numeric_range": [0.63, 1077.35]}, "C9": {"missing": 0, "unique_nonempty": 8, "numeric_range": [0.05, 1382.84]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]