File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['customer', 'demand']
Parsed column types: {'customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
customer demand
      C1     94
      C2     39
      C3     65
      C4    435
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 4}, "demand": {"missing": 0, "unique_nonempty": 4, "numeric_range": [39.0, 435.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "brewco,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "customer_demand.csv,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['Unnamed: 0', 'supply_capacity']
Parsed column types: {'Unnamed: 0': 'object', 'supply_capacity': 'int64'}
Preview only (first 10 rows):
Unnamed: 0 supply_capacity
        S1            2531
        S2              20
        S3             210
        S4             241
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 4}, "supply_capacity": {"missing": 0, "unique_nonempty": 4, "numeric_range": [20.0, 2531.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "brewco,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "customer_demand.csv,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4']
Parsed column types: {'Unnamed: 0': 'object', 'C1': 'float64', 'C2': 'float64', 'C3': 'float64', 'C4': 'float64'}
Preview only (first 10 rows):
Unnamed: 0                C1                  C2                 C3                 C4
        S1  543.756480860856  23.685276141764653 23.676386730773032 447.75143678673766
        S2 883.9151090405642 0.04977684765576961 0.0350986687216299  44.45588531711622
        S3 537.3456896658107  23.769274659075112 498.95659249465467 440.60737890439776
        S4 1791.493192397229   68.21633865655126 1432.4837339656747 1527.7635425462734
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 4}, "C1": {"missing": 0, "unique_nonempty": 4, "numeric_range": [537.3456896658107, 1791.493192397229]}, "C2": {"missing": 0, "unique_nonempty": 4, "numeric_range": [0.0497768476557696, 68.21633865655126]}, "C3": {"missing": 0, "unique_nonempty": 4, "numeric_range": [0.0350986687216299, 1432.483733965675]}, "C4": {"missing": 0, "unique_nonempty": 4, "numeric_range": [44.45588531711622, 1527.7635425462734]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "brewco,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "customer_demand.csv,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]