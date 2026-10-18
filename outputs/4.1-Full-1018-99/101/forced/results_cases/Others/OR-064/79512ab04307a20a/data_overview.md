File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 25
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
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 25}, "demand": {"missing": 0, "unique_nonempty": 23, "numeric_range": [1.0, 1097.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 24
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
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 24}, "fixed_costs": {"missing": 0, "unique_nonempty": 24, "numeric_range": [82.57, 112.65]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 24
Columns: ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25']
Parsed column types: {'Unnamed: 0': 'object', 'C1': 'float64', 'C2': 'float64', 'C3': 'float64', 'C4': 'float64', 'C5': 'float64', 'C6': 'float64', 'C7': 'float64', 'C8': 'float64', 'C9': 'float64', 'C10': 'float64', 'C11': 'float64', 'C12': 'float64', 'C13': 'float64', 'C14': 'float64', 'C15': 'float64', 'C16': 'float64', 'C17': 'float64', 'C18': 'float64', 'C19': 'float64', 'C20': 'float64', 'C21': 'float64', 'C22': 'float64', 'C23': 'float64', 'C24': 'float64', 'C25': 'float64'}
Preview only (first 10 rows):
Unnamed: 0     C1      C2                C3      C4      C5                  C6                C7      C8      C9               C10               C11               C12     C13               C14                C15     C16               C17               C18     C19               C20               C21     C22               C23               C24               C25
        S1 284.11   53.78             10.62  111.27   158.5   8.789999999999999             53.79    8.84 1911.43 8.869999999999999           1129.47            185.53  319.81             79.72             185.84  210.58             79.72 8.869999999999999    8.81              5.07 8.789999999999999 2032.76           1129.44 79.79000000000001            191.46
        S2   7.19 1031.96             90.94  276.97    0.45                 0.2             49.14    1.05 2079.54              1.45             49.14              0.05  497.49           1542.16               1.15  368.32 85.68000000000001              0.09     0.5              4.08                 0 1828.82             49.14           1543.43             72.39
        S3  151.1  884.48              4.33  277.04    0.33                0.19             49.14    0.99   99.03              1.63            884.47              0.96  497.55           1542.22               1.28   20.47           1799.25              0.09    0.02             85.83                 0 1828.87             49.14             85.75              4.02
        S4 144.16  868.75              94.2  285.48   16.93  0.9399999999999999            868.78    16.6   98.69             19.74            868.74             19.85   504.7           1816.02               0.95  433.81           1816.01              0.99   19.71             96.16              0.95   102.5             868.7           1557.86             56.07
        S5 151.34 1030.88 91.43000000000001   13.24    0.72                0.87             49.09    0.01   99.05              0.84             883.6              0.58   23.74             85.73 0.9399999999999999  430.67           1543.13              0.08    0.62             74.45                 0 2134.66 883.5700000000001              85.8 71.90000000000001
        S6   7.18   49.13             90.72  277.57    0.37                0.58           1031.74    0.76 1782.98              1.06 884.3099999999999              0.34   23.72             85.69 0.6899999999999999  368.87             85.69              0.06    0.02              4.11              0.65  101.61           1031.65           1543.73             84.63
        S7 104.38 1324.35           1829.39 1857.57 1782.69             2079.47           1324.31 2080.29       0           2080.99           1545.08 99.06999999999999 1653.07            816.75            1783.53 1687.72 700.0700000000001             99.12 1782.78           1728.93             99.03  604.42             73.58            816.05           2026.83
        S8 129.51 1031.96              4.33  276.97    0.02                0.23 884.5599999999999    1.22 2079.54              1.69             49.14              0.05  426.42 85.68000000000001               1.34  429.71 85.68000000000001              1.89     0.5 73.51000000000001                 0   101.6            884.49           1543.43              4.02
        S9  50.93    5.75           1057.85   58.62   47.63             1000.41            103.48    47.6 1642.85             47.59              5.75 999.9400000000001 1201.65            2420.6              47.61 1116.98           2420.61           1000.29  857.22              51.1           1000.61 2068.65            120.63             115.2 943.9400000000001
       S10 129.62  884.35 91.09999999999999  277.12    0.27 0.07000000000000001           1031.78    0.91   99.03              0.08             49.13              0.04  426.53             85.69               1.27  429.82             85.69               1.9    0.02              4.09              0.01 2133.83            1031.7 85.76000000000001              4.01
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 24}, "C1": {"missing": 0, "unique_nonempty": 24, "numeric_range": [7.18, 1728.02]}, "C2": {"missing": 0, "unique_nonempty": 23, "numeric_range": [0.0, 2021.48]}, "C3": {"missing": 0, "unique_nonempty": 23, "numeric_range": [4.33, 1829.39]}, "C4": {"missing": 0, "unique_nonempty": 22, "numeric_range": [9.98, 1857.57]}, "C5": {"missing": 0, "unique_nonempty": 24, "numeric_range": [0.02, 1782.69]}, "C6": {"missing": 0, "unique_nonempty": 24, "numeric_range": [0.07, 2079.47]}, "C7": {"missing": 0, "unique_nonempty": 23, "numeric_range": [0.03, 1535.55]}, "C8": {"missing": 0, "unique_nonempty": 24, "numeric_range": [0.01, 2080.29]}, "C9": {"missing": 0, "unique_nonempty": 22, "numeric_range": [0.0, 2080.31]}, "C10": {"missing": 0, "unique_nonempty": 23, "numeric_range": [0.08, 2080.99]}, "C11": {"missing": 0, "unique_nonempty": 23, "numeric_range": [0.08, 1545.08]}, "C12": {"missing": 0, "unique_nonempty": 23, "numeric_range": [0.04, 1798.36]}, "C13": {"missing": 0, "unique_nonempty": 23, "numeric_range": [20.07, 1653.07]}, "C14": {"missing": 0, "unique_nonempty": 23, "numeric_range": [1.79, 2420.6]}, "C15": {"missing": 0, "unique_nonempty": 24, "numeric_range": [0.44, 1798.43]}, "C16": {"missing": 0, "unique_nonempty": 24, "numeric_range": [16.41, 1687.72]}, "C17": {"missing": 0, "unique_nonempty": 20, "numeric_range": [0.1, 2420.61]}, "C18": {"missing": 0, "unique_nonempty": 22, "numeric_range": [0.06, 1541.17]}, "C19": {"missing": 0, "unique_nonempty": 20, "numeric_range": [0.02, 1782.78]}, "C20": {"missing": 0, "unique_nonempty": 23, "numeric_range": [4.08, 1728.93]}, "C21": {"missing": 0, "unique_nonempty": 21, "numeric_range": [0.0, 1540.87]}, "C22": {"missing": 0, "unique_nonempty": 24, "numeric_range": [97.75, 2571.8]}, "C23": {"missing": 0, "unique_nonempty": 23, "numeric_range": [0.14, 1535.48]}, "C24": {"missing": 0, "unique_nonempty": 23, "numeric_range": [2.68, 2018.83]}, "C25": {"missing": 0, "unique_nonempty": 23, "numeric_range": [3.3, 2026.83]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]