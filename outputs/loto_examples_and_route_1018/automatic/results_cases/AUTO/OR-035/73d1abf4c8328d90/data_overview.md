File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 1
Columns: ['Capacity']
Parsed column types: {'Capacity': 'int64'}
Preview only (first 10 rows):
Capacity
     180
Full-file column statistics: {"Capacity": {"missing": 0, "unique_nonempty": 1, "numeric_range": [180.0, 180.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'object', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
   ProductName Value Weight
      Baguette   888      4
     Croissant   134      2
     Sourdough   129      4
     Rye Bread   370      3
       Brioche   921      2
      Focaccia   765      1
      Ciabatta   154      2
          Pita   837      1
         Bagel   584      3
English Muffin   365      3
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 10}, "Value": {"missing": 0, "unique_nonempty": 10, "numeric_range": [129.0, 921.0]}, "Weight": {"missing": 0, "unique_nonempty": 4, "numeric_range": [1.0, 4.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}]