File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 481
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
      Product Name  Revenue Demand Initial Inventory
 ACCESSORY_1024.58  1024.58      8                40
 ACCESSORY_1025.42  1025.42     16                80
ACCESSORY_102550.0 102550.0      4                20
ACCESSORY_10347.46 10347.46      4                20
 ACCESSORY_10432.2  10432.2      6                30
ACCESSORY_10508.48 10508.48     12                60
 ACCESSORY_10932.2  10932.2      4                20
 ACCESSORY_1288.14  1288.14      4                20
 ACCESSORY_1322.04  1322.04      4                20
 ACCESSORY_1363.56  1363.56      6                30
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 481}, "Revenue": {"missing": 0, "unique_nonempty": 458, "numeric_range": [410.71, 104767.86]}, "Demand": {"missing": 0, "unique_nonempty": 78, "numeric_range": [2.0, 326.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 70, "numeric_range": [10.0, 2000.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tablet", "matching_columns": [{"column": "Product Name", "exact": 0, "prefix": 27, "contains": 27, "examples": ["TABLET_10084.74", "TABLET_12211.86", "TABLET_14669.5"]}], "exact_matching_columns": 0}]