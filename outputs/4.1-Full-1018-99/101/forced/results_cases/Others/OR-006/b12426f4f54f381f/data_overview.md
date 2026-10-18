File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['customer', 'demand']
Parsed column types: {'customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
customer demand
      C1     45
      C2     23
      C3     94
      C4     92
      C5     57
      C6     52
      C7     23
      C8     99
      C9     99
     C10     77
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 10}, "demand": {"missing": 0, "unique_nonempty": 8, "numeric_range": [23.0, 99.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['Unnamed: 0', 'supply_capacity']
Parsed column types: {'Unnamed: 0': 'object', 'supply_capacity': 'int64'}
Preview only (first 10 rows):
Unnamed: 0 supply_capacity
        S1             127
        S2             236
        S3             168
        S4             115
        S5             280
        S6             179
        S7             135
        S8             263
        S9             283
       S10             476
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 10}, "supply_capacity": {"missing": 0, "unique_nonempty": 10, "numeric_range": [115.0, 476.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
Parsed column types: {'Unnamed: 0': 'object', 'C1': 'float64', 'C2': 'float64', 'C3': 'float64', 'C4': 'float64', 'C5': 'float64', 'C6': 'float64', 'C7': 'float64', 'C8': 'float64', 'C9': 'float64', 'C10': 'float64'}
Preview only (first 10 rows):
Unnamed: 0                 C1                 C2                 C3                 C4                 C5                 C6                 C7                 C8                 C9                C10
        S1  2077.058672521021                0.0  54.33526480458508                0.0                0.0  36.17284629162332                0.0                0.0 169.33026926588778                0.0
        S2  2077.058672521021                0.0 1141.0405608962865                0.0                0.0  651.1112332492198                0.0                0.0  8.063346155518467                0.0
        S3   79.9210295982608 474.24509131006675 1477.0676289106607 22.583099586193658 474.24509131006675 41.106596962251096 474.24509131006675 474.24509131006675   624.162539502301 474.24509131006675
        S4  1659.336929105112  57.20541468776147  186.1519048103841 1201.3137084429907 1029.6974643797064  41.82210594495074  57.20541468776147 1201.3137084429907  884.5633870657458 1029.6974643797064
        S5 1297.2567040858307  77.76629131320436  24.26760227579214 1399.7932436376784  77.76629131320436  53.91161728496604 1399.7932436376784  77.76629131320436  1255.115148013589 1399.7932436376784
        S6 1998.9090658724567  985.3165435695341 2.8541686885814643   1149.53596749779  985.3165435695341  730.6923647662475  54.73980797608523  985.3165435695341 46.803102206463265   1149.53596749779
        S7 1780.3360050180179                0.0 1141.0405608962865                0.0                0.0  36.17284629162332                0.0                0.0  8.063346155518467                0.0
        S8  75.40935896233042 1338.1987290721909 21.391345987195333  74.34437383734394  74.34437383734394  937.3506239061463 1338.1987290721909 1338.1987290721909 1392.1186581383768 1338.1987290721909
        S9  98.90755583433433                0.0  978.0347664825314                0.0                0.0  651.1112332492198                0.0                0.0 169.33026926588778                0.0
       S10  2077.058672521021                0.0  54.33526480458508                0.0                0.0  36.17284629162332                0.0                0.0  145.1402307993324                0.0
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 10}, "C1": {"missing": 0, "unique_nonempty": 8, "numeric_range": [75.40935896233042, 2077.058672521021]}, "C2": {"missing": 0, "unique_nonempty": 6, "numeric_range": [0.0, 1338.1987290721909]}, "C3": {"missing": 0, "unique_nonempty": 8, "numeric_range": [2.8541686885814643, 1477.0676289106607]}, "C4": {"missing": 0, "unique_nonempty": 6, "numeric_range": [0.0, 1399.7932436376784]}, "C5": {"missing": 0, "unique_nonempty": 6, "numeric_range": [0.0, 1029.6974643797064]}, "C6": {"missing": 0, "unique_nonempty": 7, "numeric_range": [36.17284629162332, 937.3506239061464]}, "C7": {"missing": 0, "unique_nonempty": 6, "numeric_range": [0.0, 1399.7932436376784]}, "C8": {"missing": 0, "unique_nonempty": 6, "numeric_range": [0.0, 1338.1987290721909]}, "C9": {"missing": 0, "unique_nonempty": 8, "numeric_range": [8.063346155518467, 1392.1186581383768]}, "C10": {"missing": 0, "unique_nonempty": 6, "numeric_range": [0.0, 1399.7932436376784]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]