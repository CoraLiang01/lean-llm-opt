File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 35
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'int64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
  Product Name Revenue Demand Initial Inventory
   Beauty - 25      25    240              1570
   Beauty - 30      30    202              1330
  Beauty - 300     300    216              1420
   Beauty - 50      50    263              1700
  Beauty - 500     500    256              1690
 Clothing - 25      25    281              1840
 Clothing - 30      30    261              1710
Clothing - 300     300    295              1930
 Clothing - 50      50    290              1890
Clothing - 500     500    244              1570
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 35}, "Revenue": {"missing": 0, "unique_nonempty": 5, "numeric_range": [25.0, 500.0]}, "Demand": {"missing": 0, "unique_nonempty": 35, "numeric_range": [195.0, 295.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 33, "numeric_range": [1268.0, 1930.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]