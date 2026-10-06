File: /Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 131
Columns: ['option', 'tech', 'ArchiveRevisionCount', 'gen_per_lot', 'cost_per_lot', 'ArchiveFolder']
Parsed column types: {'option': 'object', 'tech': 'object', 'ArchiveRevisionCount': 'int64', 'gen_per_lot': 'int64', 'cost_per_lot': 'float64', 'ArchiveFolder': 'object'}
Preview only (first 10 rows):
  option tech ArchiveRevisionCount gen_per_lot cost_per_lot ArchiveFolder
coal_001 coal                   20          45        37.62      Folder_A
coal_002 coal                   30          55        44.43      Folder_A
coal_003 coal                   22          45        34.76      Folder_C
coal_004 coal                   22          40        33.17      Folder_B
coal_005 coal                   12          50        40.83      Folder_A
coal_006 coal                    3          40         33.5      Folder_C
coal_007 coal                   21          55        42.73      Folder_A
coal_008 coal                    8          45        34.86      Folder_B
coal_009 coal                   13          45        36.09      Folder_B
coal_010 coal                   14          50        39.16      Folder_B
Full-file column statistics: {"option": {"missing": 0, "unique_nonempty": 131}, "tech": {"missing": 0, "unique_nonempty": 3}, "ArchiveRevisionCount": {"missing": 0, "unique_nonempty": 30, "numeric_range": [1.0, 30.0]}, "gen_per_lot": {"missing": 0, "unique_nonempty": 10, "numeric_range": [15.0, 60.0]}, "cost_per_lot": {"missing": 0, "unique_nonempty": 127, "numeric_range": [18.04, 49.81]}, "ArchiveFolder": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "coal", "matching_columns": [{"column": "option", "exact": 0, "prefix": 34, "contains": 34, "examples": ["coal_001", "coal_002", "coal_003"]}, {"column": "tech", "exact": 34, "prefix": 34, "contains": 34, "examples": ["coal"]}], "exact_matching_columns": 1}, {"term": "gas", "matching_columns": [{"column": "option", "exact": 0, "prefix": 56, "contains": 56, "examples": ["gas_001", "gas_002", "gas_003"]}, {"column": "tech", "exact": 56, "prefix": 56, "contains": 56, "examples": ["gas"]}], "exact_matching_columns": 1}, {"term": "lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "renewables", "matching_columns": [{"column": "option", "exact": 0, "prefix": 41, "contains": 41, "examples": ["renewables_001", "renewables_002", "renewables_003"]}, {"column": "tech", "exact": 41, "prefix": 41, "contains": 41, "examples": ["renewables"]}], "exact_matching_columns": 1}]