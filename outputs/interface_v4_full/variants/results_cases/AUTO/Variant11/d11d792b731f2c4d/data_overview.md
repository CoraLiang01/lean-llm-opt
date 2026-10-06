File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant11/inputs/monthly_lot_sizing.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 12
Columns: ['Month', 'Demand', 'ProductionCost', 'SetupCost', 'HoldingCost', 'ProductionCapacity']
Parsed column types: {'Month': 'object', 'Demand': 'int64', 'ProductionCost': 'int64', 'SetupCost': 'int64', 'HoldingCost': 'float64', 'ProductionCapacity': 'int64'}
Preview only (first 10 rows):
Month Demand ProductionCost SetupCost HoldingCost ProductionCapacity
  M01     95             17       760         1.0                310
  M02     70             18       820         1.1                260
  M03    130             20       930         1.2                320
  M04     85             19       780         1.0                300
  M05    115             21       960         1.3                340
  M06    140             22      1040         1.4                360
  M07     90             18       800         1.1                290
  M08    125             20       900         1.2                330
  M09     75             17       740         1.0                280
  M10    150             23      1100         1.5                370
Full-file column statistics: {"Month": {"missing": 0, "unique_nonempty": 12}, "Demand": {"missing": 0, "unique_nonempty": 12, "numeric_range": [70.0, 150.0]}, "ProductionCost": {"missing": 0, "unique_nonempty": 7, "numeric_range": [17.0, 23.0]}, "SetupCost": {"missing": 0, "unique_nonempty": 12, "numeric_range": [740.0, 1100.0]}, "HoldingCost": {"missing": 0, "unique_nonempty": 7, "numeric_range": [0.0, 1.5]}, "ProductionCapacity": {"missing": 0, "unique_nonempty": 11, "numeric_range": [260.0, 370.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "month", "matching_columns": [], "exact_matching_columns": 0}]