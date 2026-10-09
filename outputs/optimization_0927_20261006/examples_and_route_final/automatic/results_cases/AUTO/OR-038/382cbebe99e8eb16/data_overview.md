File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['VehicleID', 'VehicleType', 'Capacity']
Parsed column types: {'VehicleID': 'int64', 'VehicleType': 'object', 'Capacity': 'int64'}
Preview only (first 10 rows):
VehicleID       VehicleType Capacity
        1            Sedans      100
        2              SUVs       80
        3 Electric Vehicles      120
        4   Hybrid Vehicles       90
        5            Trucks       50
        6       Sports Cars       30
        7      Compact Cars      110
        8     Luxury Sedans       40
        9              Vans       60
       10     Pickup Trucks       35
Full-file column statistics: {"VehicleID": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1.0, 10.0]}, "VehicleType": {"missing": 0, "unique_nonempty": 10}, "Capacity": {"missing": 0, "unique_nonempty": 10, "numeric_range": [30.0, 120.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "electric vehicles", "matching_columns": [{"column": "VehicleType", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Electric Vehicles"]}], "exact_matching_columns": 1}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "sedans", "matching_columns": [{"column": "VehicleType", "exact": 1, "prefix": 1, "contains": 2, "examples": ["Sedans", "Luxury Sedans"]}], "exact_matching_columns": 1}, {"term": "suvs", "matching_columns": [{"column": "VehicleType", "exact": 1, "prefix": 1, "contains": 1, "examples": ["SUVs"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv
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
Query-name evidence: [{"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "electric vehicles", "matching_columns": [{"column": "ProductName", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Electric Vehicles"]}], "exact_matching_columns": 1}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "sedans", "matching_columns": [{"column": "ProductName", "exact": 1, "prefix": 1, "contains": 2, "examples": ["Sedans", "Luxury Sedans"]}], "exact_matching_columns": 1}, {"term": "suvs", "matching_columns": [{"column": "ProductName", "exact": 1, "prefix": 1, "contains": 1, "examples": ["SUVs"]}], "exact_matching_columns": 1}]