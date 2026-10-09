File: /Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others3/value.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 140
Columns: ['item', 'value', 'ArchiveRevisionCount', 'weight', 'ArchiveFolder']
Parsed column types: {'item': 'int64', 'value': 'int64', 'ArchiveRevisionCount': 'int64', 'weight': 'int64', 'ArchiveFolder': 'object'}
Preview only (first 10 rows):
item value ArchiveRevisionCount weight ArchiveFolder
   1    10                   22      2      Folder_A
   2    40                   29      5      Folder_C
   3    30                    6      4      Folder_A
   4    50                    2      8      Folder_A
   5    35                   25      7      Folder_B
   6     6                    5      1      Folder_B
   7    33                   21      8      Folder_A
   8    57                   18      7      Folder_C
   9    42                   27      5      Folder_B
  10    24                   21      5      Folder_C
Full-file column statistics: {"item": {"missing": 0, "unique_nonempty": 140, "numeric_range": [1.0, 140.0]}, "value": {"missing": 0, "unique_nonempty": 62, "numeric_range": [4.0, 82.0]}, "ArchiveRevisionCount": {"missing": 0, "unique_nonempty": 30, "numeric_range": [1.0, 30.0]}, "weight": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1.0, 10.0]}, "ArchiveFolder": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "value", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [], "exact_matching_columns": 0}]