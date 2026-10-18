File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['PlatformId', 'Capacity']
Parsed column types: {'PlatformId': 'int64', 'Capacity': 'int64'}
Preview only (first 10 rows):
PlatformId Capacity
         1     1336
         2     1754
         3     1617
         4     1119
         5     1410
         6      627
         7      748
         8     1540
         9     1292
        10     1138
Full-file column statistics: {"PlatformId": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1.0, 10.0]}, "Capacity": {"missing": 0, "unique_nonempty": 10, "numeric_range": [627.0, 1754.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 15
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'object', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
ProductName Value Weight
     Racing    28    393
     Sports    69    195
     Action    20    192
  Adventure    62    155
        RPG    58    500
    Shooter    11    156
   Strategy    73    317
 Simulation    43    694
     Puzzle    28    751
   Fighting    57    467
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 15}, "Value": {"missing": 0, "unique_nonempty": 14, "numeric_range": [11.0, 92.0]}, "Weight": {"missing": 0, "unique_nonempty": 15, "numeric_range": [146.0, 796.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "racing", "matching_columns": [{"column": "ProductName", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Racing"]}], "exact_matching_columns": 1}, {"term": "sports", "matching_columns": [{"column": "ProductName", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Sports"]}], "exact_matching_columns": 1}, {"term": "value", "matching_columns": [], "exact_matching_columns": 0}]