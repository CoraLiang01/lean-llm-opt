File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/plant_capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 3
Columns: ['Plant', 'SupplyCapacity']
Parsed column types: {'Plant': 'object', 'SupplyCapacity': 'int64'}
Preview only (first 10 rows):
Plant SupplyCapacity
   P1            190
   P2            160
   P3            150
Full-file column statistics: {"Plant": {"missing": 0, "unique_nonempty": 3}, "SupplyCapacity": {"missing": 0, "unique_nonempty": 3, "numeric_range": [150.0, 190.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "plant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/retailer_demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['Retailer', 'Demand']
Parsed column types: {'Retailer': 'object', 'Demand': 'int64'}
Preview only (first 10 rows):
Retailer Demand
      R1     70
      R2     85
      R3     75
      R4     65
      R5     95
      R6     80
Full-file column statistics: {"Retailer": {"missing": 0, "unique_nonempty": 6}, "Demand": {"missing": 0, "unique_nonempty": 6, "numeric_range": [65.0, 95.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "retailer", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_variable_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 3
Columns: ['Plant', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']
Parsed column types: {'Plant': 'object', 'R1': 'int64', 'R2': 'int64', 'R3': 'int64', 'R4': 'int64', 'R5': 'int64', 'R6': 'int64'}
Preview only (first 10 rows):
Plant R1 R2 R3 R4 R5 R6
   P1  3  4 18 20 22 19
   P2 17 16  4  5 20 18
   P3 21 19 18 17  3  4
Full-file column statistics: {"Plant": {"missing": 0, "unique_nonempty": 3}, "R1": {"missing": 0, "unique_nonempty": 3, "numeric_range": [3.0, 21.0]}, "R2": {"missing": 0, "unique_nonempty": 3, "numeric_range": [4.0, 19.0]}, "R3": {"missing": 0, "unique_nonempty": 2, "numeric_range": [4.0, 18.0]}, "R4": {"missing": 0, "unique_nonempty": 3, "numeric_range": [5.0, 20.0]}, "R5": {"missing": 0, "unique_nonempty": 3, "numeric_range": [3.0, 22.0]}, "R6": {"missing": 0, "unique_nonempty": 3, "numeric_range": [4.0, 19.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "plant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_fixed_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 3
Columns: ['Plant', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']
Parsed column types: {'Plant': 'object', 'R1': 'int64', 'R2': 'int64', 'R3': 'int64', 'R4': 'int64', 'R5': 'int64', 'R6': 'int64'}
Preview only (first 10 rows):
Plant  R1  R2  R3  R4  R5  R6
   P1 260 300 620 670 710 660
   P2 610 590 280 310 690 640
   P3 700 650 630 600 250 290
Full-file column statistics: {"Plant": {"missing": 0, "unique_nonempty": 3}, "R1": {"missing": 0, "unique_nonempty": 3, "numeric_range": [260.0, 700.0]}, "R2": {"missing": 0, "unique_nonempty": 3, "numeric_range": [300.0, 650.0]}, "R3": {"missing": 0, "unique_nonempty": 3, "numeric_range": [280.0, 630.0]}, "R4": {"missing": 0, "unique_nonempty": 3, "numeric_range": [310.0, 670.0]}, "R5": {"missing": 0, "unique_nonempty": 3, "numeric_range": [250.0, 710.0]}, "R6": {"missing": 0, "unique_nonempty": 3, "numeric_range": [290.0, 660.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "plant", "matching_columns": [], "exact_matching_columns": 0}]