File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 14
Columns: ['DisplayID', 'Capacity']
Parsed column types: {'DisplayID': 'int64', 'Capacity': 'int64'}
Preview only (first 10 rows):
DisplayID Capacity
        1      457
        2      604
        3      751
        4      468
        5      343
        6      408
        7      741
        8      914
        9      682
       10      409
Full-file column statistics: {"DisplayID": {"missing": 0, "unique_nonempty": 14, "numeric_range": [1.0, 14.0]}, "Capacity": {"missing": 0, "unique_nonempty": 14, "numeric_range": [342.0, 914.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 20
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'object', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
 ProductName Value Weight
   Speedboat 29664     18
Fishing Boat 31778     36
   Catamaran 73501     25
       Yacht 78255     16
    Sailboat 93606     97
       Kayak 46983     35
       Canoe 95026     32
   Houseboat 57685    100
     Pontoon 60323     43
     Jet Ski 91224     15
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 20}, "Value": {"missing": 0, "unique_nonempty": 20, "numeric_range": [29664.0, 95026.0]}, "Weight": {"missing": 0, "unique_nonempty": 20, "numeric_range": [13.0, 100.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "value", "matching_columns": [], "exact_matching_columns": 0}]