File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 1
Columns: ['Capacity']
Parsed column types: {'Capacity': 'int64'}
Preview only (first 10 rows):
Capacity
    1576
Full-file column statistics: {"Capacity": {"missing": 0, "unique_nonempty": 1, "numeric_range": [1576.0, 1576.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 25
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'object', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
  ProductName Value Weight
        Sedan  1752     15
          SUV  1856     87
        Truck  8372     36
  Convertible  6168     30
      Minivan  9681     33
        Coupe  8062     72
    Hatchback  3895     75
Station Wagon  3254     71
 Electric Car  1701     51
   Hybrid Car  6799     21
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 25}, "Value": {"missing": 0, "unique_nonempty": 25, "numeric_range": [1343.0, 9681.0]}, "Weight": {"missing": 0, "unique_nonempty": 24, "numeric_range": [15.0, 97.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}]