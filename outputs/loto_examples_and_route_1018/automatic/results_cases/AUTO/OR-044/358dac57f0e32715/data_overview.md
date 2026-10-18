File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['SectionID', 'Capacity']
Parsed column types: {'SectionID': 'int64', 'Capacity': 'int64'}
Preview only (first 10 rows):
SectionID Capacity
        1      100
        2      150
        3      120
        4      130
        5       90
        6      110
        7      160
        8      140
Full-file column statistics: {"SectionID": {"missing": 0, "unique_nonempty": 8, "numeric_range": [1.0, 8.0]}, "Capacity": {"missing": 0, "unique_nonempty": 8, "numeric_range": [90.0, 160.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'int64', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
ProductName Value Weight
          1    10      2
          2    15      3
          3     8      1
          4    12      2
          5    20      4
          6    25      5
          7     5      1
          8    30      6
          9    18      3
         10    22      4
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1.0, 10.0]}, "Value": {"missing": 0, "unique_nonempty": 10, "numeric_range": [5.0, 30.0]}, "Weight": {"missing": 0, "unique_nonempty": 6, "numeric_range": [1.0, 6.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}]