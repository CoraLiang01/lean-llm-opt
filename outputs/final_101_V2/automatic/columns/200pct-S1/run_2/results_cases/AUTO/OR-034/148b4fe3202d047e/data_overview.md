File: /Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Others_example/Others3/value.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 140
Columns: ['item', 'ArchiveAccessRoute', 'value', 'ArchiveAttachmentCount', 'ArchiveRevisionCount', 'weight', 'ArchiveReviewDesk', 'ArchivePageCount', 'ArchiveFolder']
Parsed column types: {'item': 'int64', 'ArchiveAccessRoute': 'object', 'value': 'int64', 'ArchiveAttachmentCount': 'int64', 'ArchiveRevisionCount': 'int64', 'weight': 'int64', 'ArchiveReviewDesk': 'object', 'ArchivePageCount': 'int64', 'ArchiveFolder': 'object'}
Preview only (first 10 rows):
item ArchiveAccessRoute value ArchiveAttachmentCount ArchiveRevisionCount weight ArchiveReviewDesk ArchivePageCount ArchiveFolder
   1             Portal    10                     19                   22      2            Desk_A                4      Folder_A
   2       CatalogIndex    40                     26                   29      5            Desk_A               29      Folder_C
   3         LocalIndex    30                     30                    6      4            Desk_B               10      Folder_A
   4         LocalIndex    50                     24                    2      8            Desk_B               16      Folder_A
   5         LocalIndex    35                     12                   25      7            Desk_B                4      Folder_B
   6             Portal     6                     19                    5      1            Desk_C               16      Folder_B
   7       CatalogIndex    33                      9                   21      8            Desk_B               28      Folder_A
   8       CatalogIndex    57                     22                   18      7            Desk_C                7      Folder_C
   9         LocalIndex    42                     29                   27      5            Desk_C               25      Folder_B
  10         LocalIndex    24                     22                   21      5            Desk_B               26      Folder_C
Full-file column statistics: {"item": {"missing": 0, "unique_nonempty": 140, "numeric_range": [1.0, 140.0]}, "ArchiveAccessRoute": {"missing": 0, "unique_nonempty": 3}, "value": {"missing": 0, "unique_nonempty": 62, "numeric_range": [4.0, 82.0]}, "ArchiveAttachmentCount": {"missing": 0, "unique_nonempty": 30, "numeric_range": [1.0, 30.0]}, "ArchiveRevisionCount": {"missing": 0, "unique_nonempty": 30, "numeric_range": [1.0, 30.0]}, "weight": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1.0, 10.0]}, "ArchiveReviewDesk": {"missing": 0, "unique_nonempty": 3}, "ArchivePageCount": {"missing": 0, "unique_nonempty": 30, "numeric_range": [1.0, 30.0]}, "ArchiveFolder": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "value", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [], "exact_matching_columns": 0}]