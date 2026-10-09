File: /Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others3/value.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 140
Columns: ['item', 'value', 'value_previous_season', 'weight', 'weight_previous_packaging', 'category_previous_season']
Parsed column types: {'item': 'int64', 'value': 'int64', 'value_previous_season': 'int64', 'weight': 'int64', 'weight_previous_packaging': 'int64', 'category_previous_season': 'object'}
Preview only (first 10 rows):
item value value_previous_season weight weight_previous_packaging category_previous_season
   1    10                     9      2                         3                Equipment
   2    40                    44      5                         6                Household
   3    30                    29      4                         5                 Supplies
   4    50                    58      8                         7                 Supplies
   5    35                    34      7                         8                 Supplies
   6     6                     7      1                         2                 Supplies
   7    33                    34      8                         7                 Supplies
   8    57                    51      7                         8                Equipment
   9    42                    41      5                         6                Household
  10    24                    27      5                         6                Household
Full-file column statistics: {"item": {"missing": 0, "unique_nonempty": 140, "numeric_range": [1.0, 140.0]}, "value": {"missing": 0, "unique_nonempty": 62, "numeric_range": [4.0, 82.0]}, "value_previous_season": {"missing": 0, "unique_nonempty": 63, "numeric_range": [5.0, 82.0]}, "weight": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1.0, 10.0]}, "weight_previous_packaging": {"missing": 0, "unique_nonempty": 10, "numeric_range": [2.0, 11.0]}, "category_previous_season": {"missing": 0, "unique_nonempty": 4}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "value", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [], "exact_matching_columns": 0}]