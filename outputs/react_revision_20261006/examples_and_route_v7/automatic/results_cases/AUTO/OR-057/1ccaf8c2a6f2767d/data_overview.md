File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 15
Columns: ['PlatformID', 'Capacity']
Parsed column types: {'PlatformID': 'int64', 'Capacity': 'int64'}
Preview only (first 10 rows):
PlatformID Capacity
         1      995
         2     1143
         3      949
         4      969
         5     1649
         6      870
         7     1064
         8      536
         9      766
        10      532
Full-file column statistics: {"PlatformID": {"missing": 0, "unique_nonempty": 15, "numeric_range": [1.0, 15.0]}, "Capacity": {"missing": 0, "unique_nonempty": 15, "numeric_range": [532.0, 1979.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 15
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'object', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
ProductName Value Weight
     Racing    59    776
     Sports    83    573
     Action    94    127
  Adventure    41    138
        RPG    96    385
    Shooter    12    263
   Strategy    83    473
 Simulation    36    387
     Puzzle    56    390
   Fighting    27    556
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 15}, "Value": {"missing": 0, "unique_nonempty": 14, "numeric_range": [12.0, 96.0]}, "Weight": {"missing": 0, "unique_nonempty": 15, "numeric_range": [127.0, 776.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "racing", "matching_columns": [{"column": "ProductName", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Racing"]}], "exact_matching_columns": 1}, {"term": "sports", "matching_columns": [{"column": "ProductName", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Sports"]}], "exact_matching_columns": 1}, {"term": "value", "matching_columns": [], "exact_matching_columns": 0}]