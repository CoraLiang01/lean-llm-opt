File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5681
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
Product Name  Revenue Demand Initial Inventory
       FDW58 107.8622     10               150
       FDW14  87.3198     20               100
       NCN55 241.7538     10               300
       FDQ58  155.034     40               100
       FDY38   234.23     30               300
       FDH56 117.1492     30                50
       FDL48  50.1034     30               200
       FDC48  81.0592     40               300
       FDN33  95.7436     20               100
       FDA36 186.8924     50               250
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 1543}, "Revenue": {"missing": 0, "unique_nonempty": 4402, "numeric_range": [31.99, 266.5884]}, "Demand": {"missing": 0, "unique_nonempty": 5, "numeric_range": [10.0, 50.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 6, "numeric_range": [50.0, 300.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fdk57", "matching_columns": [{"column": "Product Name", "exact": 8, "prefix": 8, "contains": 8, "examples": ["FDK57"]}], "exact_matching_columns": 1}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]