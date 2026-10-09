File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 7
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'int64', 'Demand': 'int64', 'Initial Inventory': 'float64'}
Preview only (first 10 rows):
   Product Name Revenue Demand Initial Inventory
       Aalopuri      20   1483           10440.0
    Cold coffee      40   1918           13610.0
        Frankie      50   1623           11500.0
       Panipuri      20   1720           12260.0
       Sandwich      60   1558           10970.0
Sugarcane juice      25   1791           12780.0
        Vadapav      20   1426           10060.0
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 7}, "Revenue": {"missing": 0, "unique_nonempty": 5, "numeric_range": [20.0, 60.0]}, "Demand": {"missing": 0, "unique_nonempty": 7, "numeric_range": [1426.0, 1918.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 7, "numeric_range": [10060.0, 13610.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "aalop", "matching_columns": [{"column": "Product Name", "exact": 0, "prefix": 1, "contains": 1, "examples": ["Aalopuri"]}], "exact_matching_columns": 0}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]