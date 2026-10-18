File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 2
Columns: ['customer', 'demand']
Parsed column types: {'customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
customer demand
      C1    144
      C2    216
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 2}, "demand": {"missing": 0, "unique_nonempty": 2, "numeric_range": [144.0, 216.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv,", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 2
Columns: ['Unnamed: 0', 'fixed_costs']
Parsed column types: {'Unnamed: 0': 'object', 'fixed_costs': 'float64'}
Preview only (first 10 rows):
Unnamed: 0 fixed_costs
        S1      105.97
        S2       85.31
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 2}, "fixed_costs": {"missing": 0, "unique_nonempty": 2, "numeric_range": [85.31, 105.97]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv,", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 2
Columns: ['Unnamed: 0', 'C1', 'C2']
Parsed column types: {'Unnamed: 0': 'object', 'C1': 'float64', 'C2': 'float64'}
Preview only (first 10 rows):
Unnamed: 0                  C1      C2
        S1             2358.39 1492.08
        S2 0.07000000000000001   52.32
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 2}, "C1": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.07, 2358.39]}, "C2": {"missing": 0, "unique_nonempty": 2, "numeric_range": [52.32, 1492.08]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv,", "matching_columns": [], "exact_matching_columns": 0}]