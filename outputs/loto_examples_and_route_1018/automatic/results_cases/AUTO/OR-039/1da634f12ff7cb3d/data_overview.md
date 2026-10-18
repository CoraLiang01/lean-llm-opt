File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['Warehouse ID', 'Capacity']
Parsed column types: {'Warehouse ID': 'object', 'Capacity': 'int64'}
Preview only (first 10 rows):
Warehouse ID Capacity
 Warehouse 1      100
 Warehouse 2       80
 Warehouse 3      120
 Warehouse 4       90
 Warehouse 5       50
 Warehouse 6       30
 Warehouse 7      110
 Warehouse 8       40
 Warehouse 9       60
Warehouse 10       35
Full-file column statistics: {"Warehouse ID": {"missing": 0, "unique_nonempty": 10}, "Capacity": {"missing": 0, "unique_nonempty": 10, "numeric_range": [30.0, 120.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'object', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
      ProductName Value Weight
           Sedans  1200     20
             SUVs  1800     15
Electric Vehicles  2500     25
  Hybrid Vehicles  2000     18
           Trucks  1500     10
      Sports Cars  3000      5
     Compact Cars  1000     22
    Luxury Sedans  3500      8
             Vans  1600     12
    Pickup Trucks  1700      7
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 10}, "Value": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1000.0, 3500.0]}, "Weight": {"missing": 0, "unique_nonempty": 10, "numeric_range": [5.0, 25.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "value", "matching_columns": [], "exact_matching_columns": 0}]