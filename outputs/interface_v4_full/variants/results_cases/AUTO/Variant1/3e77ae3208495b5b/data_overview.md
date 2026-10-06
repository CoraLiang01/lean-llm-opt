File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant1/inputs/monthly_lot_sizing.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 24
Columns: ['Month', 'Demand', 'ProductionCost', 'SetupCost', 'HoldingCost', 'ProductionCapacity']
Parsed column types: {'Month': 'object', 'Demand': 'int64', 'ProductionCost': 'int64', 'SetupCost': 'int64', 'HoldingCost': 'float64', 'ProductionCapacity': 'int64'}
Preview only (first 10 rows):
Month Demand ProductionCost SetupCost HoldingCost ProductionCapacity
  M01     80             18       900         1.1                420
  M02    120             20      1100         1.2                360
  M03     60             19       950         1.0                380
  M04    150             21      1200         1.3                420
  M05     90             18       850         1.1                350
  M06    110             22      1300         1.4                430
  M07    140             20      1000         1.2                390
  M08     70             19       900         1.0                360
  M09    160             23      1400         1.5                440
  M10    100             21      1150         1.2                400
Full-file column statistics: {"Month": {"missing": 0, "unique_nonempty": 24}, "Demand": {"missing": 0, "unique_nonempty": 21, "numeric_range": [60.0, 160.0]}, "ProductionCost": {"missing": 0, "unique_nonempty": 6, "numeric_range": [18.0, 23.0]}, "SetupCost": {"missing": 0, "unique_nonempty": 13, "numeric_range": [800.0, 1400.0]}, "HoldingCost": {"missing": 0, "unique_nonempty": 7, "numeric_range": [0.0, 1.5]}, "ProductionCapacity": {"missing": 0, "unique_nonempty": 10, "numeric_range": [350.0, 440.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "month", "matching_columns": [], "exact_matching_columns": 0}, {"term": "productioncapacity", "matching_columns": [], "exact_matching_columns": 0}]