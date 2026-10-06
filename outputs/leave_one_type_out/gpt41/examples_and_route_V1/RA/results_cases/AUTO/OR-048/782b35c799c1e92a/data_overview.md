File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 15
Columns: ['StorageID', 'Capacity']
Parsed column types: {'StorageID': 'int64', 'Capacity': 'int64'}
Preview only (first 10 rows):
StorageID Capacity
        1     1083
        2     1840
        3      770
        4     1299
        5     1259
        6      543
        7     1831
        8      855
        9      619
       10      637
Full-file column statistics: {"StorageID": {"missing": 0, "unique_nonempty": 15, "numeric_range": [1.0, 15.0]}, "Capacity": {"missing": 0, "unique_nonempty": 15, "numeric_range": [543.0, 1840.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'object', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
       ProductName Value Weight
       Window Unit  4811    114
     Portable Unit  1130    200
      Split System  1611    106
   Ductless System  3368    256
        Central AC  2135    268
         Hybrid AC  1046    185
     Geothermal AC  4030    299
          Smart AC  3761    131
Evaporative Cooler  3523    139
      Package Unit  1701    105
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 10}, "Value": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1046.0, 4811.0]}, "Weight": {"missing": 0, "unique_nonempty": 10, "numeric_range": [105.0, 299.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "value", "matching_columns": [], "exact_matching_columns": 0}]