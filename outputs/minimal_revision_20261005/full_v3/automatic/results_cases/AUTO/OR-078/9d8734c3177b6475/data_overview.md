File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 131
Columns: ['option', 'tech', 'gen_per_lot', 'cost_per_lot', 'unit_cost_est']
Parsed column types: {'option': 'object', 'tech': 'object', 'gen_per_lot': 'int64', 'cost_per_lot': 'float64', 'unit_cost_est': 'float64'}
Preview only (first 10 rows):
  option tech gen_per_lot cost_per_lot unit_cost_est
coal_001 coal          45        37.62         0.836
coal_002 coal          55        44.43        0.8078
coal_003 coal          45        34.76        0.7724
coal_004 coal          40        33.17        0.8293
coal_005 coal          50        40.83        0.8166
coal_006 coal          40         33.5        0.8375
coal_007 coal          55        42.73        0.7769
coal_008 coal          45        34.86        0.7747
coal_009 coal          45        36.09         0.802
coal_010 coal          50        39.16        0.7832
Full-file column statistics: {"option": {"missing": 0, "unique_nonempty": 131}, "tech": {"missing": 0, "unique_nonempty": 3}, "gen_per_lot": {"missing": 0, "unique_nonempty": 10, "numeric_range": [15.0, 60.0]}, "cost_per_lot": {"missing": 0, "unique_nonempty": 127, "numeric_range": [18.04, 49.81]}, "unit_cost_est": {"missing": 0, "unique_nonempty": 129, "numeric_range": [0.7638, 1.3115]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "coal", "matching_columns": [{"column": "option", "exact": 0, "prefix": 34, "contains": 34, "examples": ["coal_001", "coal_002", "coal_003"]}, {"column": "tech", "exact": 34, "prefix": 34, "contains": 34, "examples": ["coal"]}], "exact_matching_columns": 1}, {"term": "gas", "matching_columns": [{"column": "option", "exact": 0, "prefix": 56, "contains": 56, "examples": ["gas_001", "gas_002", "gas_003"]}, {"column": "tech", "exact": 56, "prefix": 56, "contains": 56, "examples": ["gas"]}], "exact_matching_columns": 1}, {"term": "lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "renewables", "matching_columns": [{"column": "option", "exact": 0, "prefix": 41, "contains": 41, "examples": ["renewables_001", "renewables_002", "renewables_003"]}, {"column": "tech", "exact": 41, "prefix": 41, "contains": 41, "examples": ["renewables"]}], "exact_matching_columns": 1}]