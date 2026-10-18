File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['BookshelfID', 'Capacity']
Parsed column types: {'BookshelfID': 'int64', 'Capacity': 'int64'}
Preview only (first 10 rows):
BookshelfID Capacity
          1      200
          2      200
          3      300
          4      400
          5      550
          6      600
          7      650
          8      750
          9      820
         10      570
Full-file column statistics: {"BookshelfID": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1.0, 10.0]}, "Capacity": {"missing": 0, "unique_nonempty": 9, "numeric_range": [200.0, 820.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 25
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'object', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
           ProductName Value Weight
      The Great Gatsby    50     10
 To Kill a Mockingbird    70     20
                  1984    30      5
   Pride and Prejudice    60     15
The Catcher in the Rye    80     25
             Moby Dick    90     30
             Jane Eyre    40     12
         War and Peace   100     35
           The Odyssey    55     10
  Crime and Punishment    75     20
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 25}, "Value": {"missing": 0, "unique_nonempty": 21, "numeric_range": [30.0, 120.0]}, "Weight": {"missing": 0, "unique_nonempty": 20, "numeric_range": [5.0, 50.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "value", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [], "exact_matching_columns": 0}]