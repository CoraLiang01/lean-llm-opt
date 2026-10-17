File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 19
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
              Product Name  Revenue Demand Initial Inventory
              20in Monitor  38.4965   8230             41290
    27in 4K Gaming Monitor 261.2933  12474             62440
          27in FHD Monitor  52.4965  15057             75500
    34in Ultrawide Monitor 254.5933  12380             61990
     AA Batteries (4-pack)     1.92  49137            276350
    AAA Batteries (4-pack)    1.495  53353            310170
  Apple Airpods Headphones     52.5  31211            156610
Bose SoundSport Headphones   49.995  26782            134570
             Flatscreen TV    201.0   9619             48190
              Google Phone    402.0  11057             55320
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 19}, "Revenue": {"missing": 0, "unique_nonempty": 17, "numeric_range": [1.495, 1139.0]}, "Demand": {"missing": 0, "unique_nonempty": 19, "numeric_range": [1292.0, 53353.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 19, "numeric_range": [6460.0, 310170.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "27in", "matching_columns": [{"column": "Product Name", "exact": 0, "prefix": 2, "contains": 2, "examples": ["27in 4K Gaming Monitor", "27in FHD Monitor"]}], "exact_matching_columns": 0}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]