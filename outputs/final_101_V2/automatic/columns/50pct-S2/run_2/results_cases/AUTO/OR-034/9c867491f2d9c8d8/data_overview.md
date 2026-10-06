File: /Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S2/Large-scale-or/Others_example/Others3/value.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 140
Columns: ['item', 'value', 'DisplayInquiryCountLastMonth', 'weight', 'MerchandisingTeam']
Parsed column types: {'item': 'int64', 'value': 'int64', 'DisplayInquiryCountLastMonth': 'int64', 'weight': 'int64', 'MerchandisingTeam': 'object'}
Preview only (first 10 rows):
item value DisplayInquiryCountLastMonth weight MerchandisingTeam
   1    10                           25      2            Team_C
   2    40                           23      5            Team_C
   3    30                           29      4            Team_C
   4    50                            4      8            Team_A
   5    35                            5      7            Team_B
   6     6                           21      1            Team_A
   7    33                            8      8            Team_C
   8    57                           12      7            Team_C
   9    42                           24      5            Team_A
  10    24                            7      5            Team_A
Full-file column statistics: {"item": {"missing": 0, "unique_nonempty": 140, "numeric_range": [1.0, 140.0]}, "value": {"missing": 0, "unique_nonempty": 62, "numeric_range": [4.0, 82.0]}, "DisplayInquiryCountLastMonth": {"missing": 0, "unique_nonempty": 30, "numeric_range": [1.0, 30.0]}, "weight": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1.0, 10.0]}, "MerchandisingTeam": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "value", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [], "exact_matching_columns": 0}]