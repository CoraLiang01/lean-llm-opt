File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 12
Columns: ['customer', 'demand']
Parsed column types: {'customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
customer demand
      C1   1097
      C2     61
      C3     11
      C4      7
      C5     82
      C6     37
      C7    483
      C8    582
      C9    223
     C10     89
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 12}, "demand": {"missing": 0, "unique_nonempty": 12, "numeric_range": [7.0, 1097.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 12
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
        S7             97.05
        S8               103
        S9             90.45
       S10             96.73
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 12}, "fixed_costs": {"missing": 0, "unique_nonempty": 12, "numeric_range": [90.45, 112.65]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 12
Columns: ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']
Parsed column types: {'Unnamed: 0': 'object', 'C1': 'float64', 'C2': 'float64', 'C3': 'float64', 'C4': 'float64', 'C5': 'float64', 'C6': 'float64', 'C7': 'float64', 'C8': 'float64', 'C9': 'float64', 'C10': 'float64', 'C11': 'float64', 'C12': 'float64'}
Preview only (first 10 rows):
Unnamed: 0     C1      C2                C3      C4      C5                  C6                C7      C8      C9               C10               C11               C12
        S1 284.11   53.78             10.62  111.27   158.5   8.789999999999999             53.79    8.84 1911.43 8.869999999999999           1129.47            185.53
        S2   7.19 1031.96             90.94  276.97    0.45                 0.2             49.14    1.05 2079.54              1.45             49.14              0.05
        S3  151.1  884.48              4.33  277.04    0.33                0.19             49.14    0.99   99.03              1.63            884.47              0.96
        S4 144.16  868.75              94.2  285.48   16.93  0.9399999999999999            868.78    16.6   98.69             19.74            868.74             19.85
        S5 151.34 1030.88 91.43000000000001   13.24    0.72                0.87             49.09    0.01   99.05              0.84             883.6              0.58
        S6   7.18   49.13             90.72  277.57    0.37                0.58           1031.74    0.76 1782.98              1.06 884.3099999999999              0.34
        S7 104.38 1324.35           1829.39 1857.57 1782.69             2079.47           1324.31 2080.29       0           2080.99           1545.08 99.06999999999999
        S8 129.51 1031.96              4.33  276.97    0.02                0.23 884.5599999999999    1.22 2079.54              1.69             49.14              0.05
        S9  50.93    5.75           1057.85   58.62   47.63             1000.41            103.48    47.6 1642.85             47.59              5.75 999.9400000000001
       S10 129.62  884.35 91.09999999999999  277.12    0.27 0.07000000000000001           1031.78    0.91   99.03              0.08             49.13              0.04
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 12}, "C1": {"missing": 0, "unique_nonempty": 12, "numeric_range": [7.18, 959.55]}, "C2": {"missing": 0, "unique_nonempty": 11, "numeric_range": [0.0, 1324.35]}, "C3": {"missing": 0, "unique_nonempty": 11, "numeric_range": [4.33, 1829.39]}, "C4": {"missing": 0, "unique_nonempty": 11, "numeric_range": [13.24, 1857.57]}, "C5": {"missing": 0, "unique_nonempty": 12, "numeric_range": [0.02, 1782.69]}, "C6": {"missing": 0, "unique_nonempty": 12, "numeric_range": [0.07, 2079.47]}, "C7": {"missing": 0, "unique_nonempty": 11, "numeric_range": [0.03, 1324.31]}, "C8": {"missing": 0, "unique_nonempty": 12, "numeric_range": [0.01, 2080.29]}, "C9": {"missing": 0, "unique_nonempty": 10, "numeric_range": [0.0, 2079.54]}, "C10": {"missing": 0, "unique_nonempty": 11, "numeric_range": [0.08, 2080.99]}, "C11": {"missing": 0, "unique_nonempty": 11, "numeric_range": [0.08, 1545.08]}, "C12": {"missing": 0, "unique_nonempty": 11, "numeric_range": [0.04, 1031.53]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]