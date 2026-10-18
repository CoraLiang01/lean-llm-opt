File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['ShelfID', 'Capacity']
Parsed column types: {'ShelfID': 'int64', 'Capacity': 'float64'}
Preview only (first 10 rows):
ShelfID Capacity
      1      5.0
      2      7.0
      3      6.0
      4      8.0
      5      5.5
      6      9.0
      7      6.5
      8      7.5
      9      8.2
     10      5.7
Full-file column statistics: {"ShelfID": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1.0, 10.0]}, "Capacity": {"missing": 0, "unique_nonempty": 10, "numeric_range": [5.0, 9.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 20
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'object', 'Value': 'int64', 'Weight': 'float64'}
Preview only (first 10 rows):
      ProductName Value Weight
       Smartphone   200    1.0
           Laptop  1500    5.0
       Headphones   100    0.5
           Camera   800    2.0
       Smartwatch   250    0.3
           Tablet   600    1.5
Bluetooth Speaker   150    1.0
         Keyboard    80    0.8
            Mouse    50    0.2
          Monitor   300    3.0
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 20}, "Value": {"missing": 0, "unique_nonempty": 19, "numeric_range": [25.0, 1500.0]}, "Weight": {"missing": 0, "unique_nonempty": 14, "numeric_range": [0.02, 5.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "value", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [], "exact_matching_columns": 0}]