File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['customer', 'demand']
Parsed column types: {'customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
customer demand
      C1    216
      C2    216
      C3    216
      C4    144
      C5    144
      C6    144
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 6}, "demand": {"missing": 0, "unique_nonempty": 2, "numeric_range": [144.0, 216.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['Unnamed: 0', 'fixed_costs']
Parsed column types: {'Unnamed: 0': 'object', 'fixed_costs': 'float64'}
Preview only (first 10 rows):
Unnamed: 0       fixed_costs
        S1             98.88
        S2             99.73
        S3 94.01000000000001
        S4             93.77
        S5            107.59
        S6            112.65
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 6}, "fixed_costs": {"missing": 0, "unique_nonempty": 6, "numeric_range": [93.77, 112.65]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
Parsed column types: {'Unnamed: 0': 'object', 'C1': 'float64', 'C2': 'float64', 'C3': 'float64', 'C4': 'float64', 'C5': 'float64', 'C6': 'float64'}
Preview only (first 10 rows):
Unnamed: 0      C1     C2                C3      C4                  C5      C6
        S1    0.08  52.33 73.56999999999999 1237.33 0.07000000000000001  112.16
        S2   46.02 175.23           2026.83  299.89              966.53 1590.42
        S3 1031.74  78.13             99.02  277.07              884.45 1800.86
        S4  868.75   94.2           1776.34  285.48              868.85   86.55
        S5    1577 760.15           2090.19    43.2             1577.12 1095.17
        S6   49.14   4.33           2079.57  277.04             1032.01 1543.49
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 6}, "C1": {"missing": 0, "unique_nonempty": 6, "numeric_range": [0.08, 1577.0]}, "C2": {"missing": 0, "unique_nonempty": 6, "numeric_range": [4.33, 760.15]}, "C3": {"missing": 0, "unique_nonempty": 6, "numeric_range": [73.57, 2090.19]}, "C4": {"missing": 0, "unique_nonempty": 6, "numeric_range": [43.2, 1237.33]}, "C5": {"missing": 0, "unique_nonempty": 6, "numeric_range": [0.07, 1577.12]}, "C6": {"missing": 0, "unique_nonempty": 6, "numeric_range": [86.55, 1800.86]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]