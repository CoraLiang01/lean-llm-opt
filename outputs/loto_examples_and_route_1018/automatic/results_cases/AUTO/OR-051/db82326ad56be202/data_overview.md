File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['CabinetID', 'Capacity']
Parsed column types: {'CabinetID': 'int64', 'Capacity': 'int64'}
Preview only (first 10 rows):
CabinetID Capacity
        1      400
        2      600
        3      500
        4      700
        5      450
        6      650
        7      550
        8      750
        9      480
       10      520
Full-file column statistics: {"CabinetID": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1.0, 10.0]}, "Capacity": {"missing": 0, "unique_nonempty": 10, "numeric_range": [400.0, 750.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 18
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'object', 'Value': 'int64', 'Weight': 'float64'}
Preview only (first 10 rows):
    ProductName Value Weight
 Espresso Beans   100    1.0
Colombian Roast   150    1.5
  Arabica Blend    80    1.2
   French Roast   120    1.3
  Italian Roast   130    1.4
    House Blend   110    1.1
 Sumatra Coffee   160    1.8
     Mocha Java    90    1.2
Hazelnut Flavor    95    1.0
  Caramel Blend   105    1.3
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 18}, "Value": {"missing": 0, "unique_nonempty": 18, "numeric_range": [60.0, 170.0]}, "Weight": {"missing": 0, "unique_nonempty": 9, "numeric_range": [1.0, 1.8]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "value", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [], "exact_matching_columns": 0}]