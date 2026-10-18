File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['ShelfID', 'Capacity']
Parsed column types: {'ShelfID': 'int64', 'Capacity': 'int64'}
Preview only (first 10 rows):
ShelfID Capacity
      1      750
      2      820
      3      570
      4      800
      5      550
      6      900
      7      650
      8      800
      9      850
     10      900
Full-file column statistics: {"ShelfID": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1.0, 10.0]}, "Capacity": {"missing": 0, "unique_nonempty": 8, "numeric_range": [550.0, 900.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 20
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'int64', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
ProductName Value Weight
          1    55     10
          2    75     20
          3    65      5
          4    60     15
          5    80     25
          6    90     35
          7    40     45
          8   100     55
          9    55     65
         10    75     20
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 20, "numeric_range": [1.0, 20.0]}, "Value": {"missing": 0, "unique_nonempty": 12, "numeric_range": [40.0, 120.0]}, "Weight": {"missing": 0, "unique_nonempty": 16, "numeric_range": [5.0, 100.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "value", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [], "exact_matching_columns": 0}]