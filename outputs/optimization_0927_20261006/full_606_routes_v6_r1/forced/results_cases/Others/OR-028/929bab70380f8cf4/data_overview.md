File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM19/WomenClothingEcommerceSalesData.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 24
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'int64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
Product Name Revenue Demand Initial Inventory
     sku_I27     238      6                30
    sku_I499     287      4                20
    sku_I719     268     16                80
     sku_T18     318     14                70
     sku_T29     207      4                20
     sku_T39     258     32               160
    sku_T499     249      8                40
      sku_T9     227      2                10
    sku_3081     198     10                50
     sku_339     254      8                40
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 24}, "Revenue": {"missing": 0, "unique_nonempty": 17, "numeric_range": [198.0, 318.0]}, "Demand": {"missing": 0, "unique_nonempty": 13, "numeric_range": [2.0, 570.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 13, "numeric_range": [10.0, 2870.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]