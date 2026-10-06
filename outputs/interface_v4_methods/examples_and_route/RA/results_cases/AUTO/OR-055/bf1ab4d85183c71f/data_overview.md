File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 14
Columns: ['DisplayID', 'Capacity']
Parsed column types: {'DisplayID': 'int64', 'Capacity': 'int64'}
Preview only (first 10 rows):
DisplayID Capacity
        1      356
        2      478
        3      305
        4      291
        5      168
        6      449
        7      139
        8      383
        9      472
       10      288
Full-file column statistics: {"DisplayID": {"missing": 0, "unique_nonempty": 14, "numeric_range": [1.0, 14.0]}, "Capacity": {"missing": 0, "unique_nonempty": 14, "numeric_range": [139.0, 478.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 20
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'object', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
 ProductName Value Weight
   Speedboat 69978     18
Fishing Boat 54011     42
   Catamaran 36352     49
       Yacht 51521     42
    Sailboat 50415     41
       Kayak 76109     48
       Canoe 50462     22
   Houseboat 28989     29
     Pontoon 23318     45
     Jet Ski 26142     14
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 20}, "Value": {"missing": 0, "unique_nonempty": 20, "numeric_range": [22839.0, 95265.0]}, "Weight": {"missing": 0, "unique_nonempty": 17, "numeric_range": [14.0, 49.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "value", "matching_columns": [], "exact_matching_columns": 0}]