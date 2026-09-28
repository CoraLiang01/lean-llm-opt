File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 100
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
Product Name Revenue Demand Initial Inventory
   bbq_ckn_l   20.75   1960              9920
   bbq_ckn_m   16.75   1884              9560
   bbq_ckn_s   12.75    963              4840
  big_meat_s      12   3729             19140
brie_carre_s   23.65    970              4900
 calabrese_l   20.25    550              2760
 calabrese_m   16.25   1116              5620
 calabrese_s   12.25    198               990
  cali_ckn_l   20.75   1823              9270
  cali_ckn_m   16.75   1858              9440
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 100}, "Revenue": {"missing": 0, "unique_nonempty": 25, "numeric_range": [9.75, 35.95]}, "Demand": {"missing": 0, "unique_nonempty": 85, "numeric_range": [56.0, 3729.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 85, "numeric_range": [280.0, 19140.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]