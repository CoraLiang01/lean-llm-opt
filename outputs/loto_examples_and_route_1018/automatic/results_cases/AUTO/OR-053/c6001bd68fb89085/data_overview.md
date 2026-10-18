File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['ShelfID', 'Capacity']
Parsed column types: {'ShelfID': 'int64', 'Capacity': 'int64'}
Preview only (first 10 rows):
ShelfID Capacity
      1      500
      2      700
      3      600
      4      800
      5      550
      6      900
      7      650
      8      750
      9      820
     10      570
Full-file column statistics: {"ShelfID": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1.0, 10.0]}, "Capacity": {"missing": 0, "unique_nonempty": 10, "numeric_range": [500.0, 900.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 20
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'int64', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
ProductName Value Weight
          1    50     10
          2    70     20
          3    30      5
          4    60     15
          5    80     25
          6    90     30
          7    40     12
          8   100     35
          9    55     10
         10    75     20
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 20, "numeric_range": [1.0, 20.0]}, "Value": {"missing": 0, "unique_nonempty": 16, "numeric_range": [30.0, 120.0]}, "Weight": {"missing": 0, "unique_nonempty": 16, "numeric_range": [5.0, 50.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "value", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [], "exact_matching_columns": 0}]