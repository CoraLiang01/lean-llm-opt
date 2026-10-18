File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 1
Columns: ['Capacity']
Parsed column types: {'Capacity': 'int64'}
Preview only (first 10 rows):
Capacity
     765
Full-file column statistics: {"Capacity": {"missing": 0, "unique_nonempty": 1, "numeric_range": [765.0, 765.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 25
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'object', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
  ProductName Value Weight
        Sedan  2524     99
          SUV  4614     55
        Truck  8416     75
  Convertible  5917     94
      Minivan  9048     80
        Coupe  1140     82
    Hatchback  8962     71
Station Wagon  1888    100
 Electric Car  8487     28
   Hybrid Car  4425     93
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 25}, "Value": {"missing": 0, "unique_nonempty": 25, "numeric_range": [1140.0, 9895.0]}, "Weight": {"missing": 0, "unique_nonempty": 22, "numeric_range": [6.0, 100.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}]