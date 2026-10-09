File: /Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 131
Columns: ['option', 'tech', 'cost_per_lot_previous_period', 'gen_per_lot', 'cost_per_lot', 'contract_class_previous_period']
Parsed column types: {'option': 'object', 'tech': 'object', 'cost_per_lot_previous_period': 'float64', 'gen_per_lot': 'int64', 'cost_per_lot': 'float64', 'contract_class_previous_period': 'object'}
Preview only (first 10 rows):
  option tech cost_per_lot_previous_period gen_per_lot cost_per_lot contract_class_previous_period
coal_001 coal                    33.436656          45        37.62                       PeakLoad
coal_002 coal                    39.373866          55        44.43                       PeakLoad
coal_003 coal                    34.614008          45        34.76                RenewableBundle
coal_004 coal                    37.797215          40        33.17                       BaseLoad
coal_005 coal                    36.698004          50        40.83                       PeakLoad
coal_006 coal                     28.11990          40         33.5                       PeakLoad
coal_007 coal                    49.284782          55        42.73                       BaseLoad
coal_008 coal                    33.716592          45        34.86                RenewableBundle
coal_009 coal                    36.191052          45        36.09                       BaseLoad
coal_010 coal                     34.81324          50        39.16                       BaseLoad
Full-file column statistics: {"option": {"missing": 0, "unique_nonempty": 131}, "tech": {"missing": 0, "unique_nonempty": 3}, "cost_per_lot_previous_period": {"missing": 0, "unique_nonempty": 131, "numeric_range": [15.043581, 54.145757]}, "gen_per_lot": {"missing": 0, "unique_nonempty": 10, "numeric_range": [15.0, 60.0]}, "cost_per_lot": {"missing": 0, "unique_nonempty": 127, "numeric_range": [18.04, 49.81]}, "contract_class_previous_period": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "coal", "matching_columns": [{"column": "option", "exact": 0, "prefix": 34, "contains": 34, "examples": ["coal_001", "coal_002", "coal_003"]}, {"column": "tech", "exact": 34, "prefix": 34, "contains": 34, "examples": ["coal"]}], "exact_matching_columns": 1}, {"term": "gas", "matching_columns": [{"column": "option", "exact": 0, "prefix": 56, "contains": 56, "examples": ["gas_001", "gas_002", "gas_003"]}, {"column": "tech", "exact": 56, "prefix": 56, "contains": 56, "examples": ["gas"]}], "exact_matching_columns": 1}, {"term": "lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "renewables", "matching_columns": [{"column": "option", "exact": 0, "prefix": 41, "contains": 41, "examples": ["renewables_001", "renewables_002", "renewables_003"]}, {"column": "tech", "exact": 41, "prefix": 41, "contains": 41, "examples": ["renewables"]}], "exact_matching_columns": 1}]