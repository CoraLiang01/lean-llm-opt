File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant23/inputs/depot_capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['Depot', 'SupplyCapacity']
Parsed column types: {'Depot': 'object', 'SupplyCapacity': 'int64'}
Preview only (first 10 rows):
Depot SupplyCapacity
   D1            120
   D2            100
   D3            140
   D4             90
Full-file column statistics: {"Depot": {"missing": 0, "unique_nonempty": 4}, "SupplyCapacity": {"missing": 0, "unique_nonempty": 4, "numeric_range": [90.0, 140.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "depot", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant23/inputs/market_demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5
Columns: ['Market', 'Demand']
Parsed column types: {'Market': 'object', 'Demand': 'int64'}
Preview only (first 10 rows):
Market Demand
    M1     65
    M2     80
    M3     75
    M4     95
    M5     70
Full-file column statistics: {"Market": {"missing": 0, "unique_nonempty": 5}, "Demand": {"missing": 0, "unique_nonempty": 5, "numeric_range": [65.0, 95.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "market", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant23/inputs/route_variable_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['Depot', 'M1', 'M2', 'M3', 'M4', 'M5']
Parsed column types: {'Depot': 'object', 'M1': 'int64', 'M2': 'int64', 'M3': 'int64', 'M4': 'int64', 'M5': 'int64'}
Preview only (first 10 rows):
Depot M1 M2 M3 M4 M5
   D1  4  5 17 18 16
   D2 15 14  3  6 17
   D3 18 16 15  4  5
   D4  6  7 14 16 13
Full-file column statistics: {"Depot": {"missing": 0, "unique_nonempty": 4}, "M1": {"missing": 0, "unique_nonempty": 4, "numeric_range": [4.0, 18.0]}, "M2": {"missing": 0, "unique_nonempty": 4, "numeric_range": [5.0, 16.0]}, "M3": {"missing": 0, "unique_nonempty": 4, "numeric_range": [3.0, 17.0]}, "M4": {"missing": 0, "unique_nonempty": 4, "numeric_range": [4.0, 18.0]}, "M5": {"missing": 0, "unique_nonempty": 4, "numeric_range": [5.0, 17.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "depot", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant23/inputs/route_fixed_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['Depot', 'M1', 'M2', 'M3', 'M4', 'M5']
Parsed column types: {'Depot': 'object', 'M1': 'int64', 'M2': 'int64', 'M3': 'int64', 'M4': 'int64', 'M5': 'int64'}
Preview only (first 10 rows):
Depot  M1  M2  M3  M4  M5
   D1 240 270 560 590 540
   D2 520 500 230 280 570
   D3 610 580 540 260 250
   D4 290 320 500 550 480
Full-file column statistics: {"Depot": {"missing": 0, "unique_nonempty": 4}, "M1": {"missing": 0, "unique_nonempty": 4, "numeric_range": [240.0, 610.0]}, "M2": {"missing": 0, "unique_nonempty": 4, "numeric_range": [270.0, 580.0]}, "M3": {"missing": 0, "unique_nonempty": 4, "numeric_range": [230.0, 560.0]}, "M4": {"missing": 0, "unique_nonempty": 4, "numeric_range": [260.0, 590.0]}, "M5": {"missing": 0, "unique_nonempty": 4, "numeric_range": [250.0, 570.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "depot", "matching_columns": [], "exact_matching_columns": 0}]