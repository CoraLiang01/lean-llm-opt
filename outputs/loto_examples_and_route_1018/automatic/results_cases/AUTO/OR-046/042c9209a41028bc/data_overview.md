File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 1
Columns: ['Capacity']
Parsed column types: {'Capacity': 'int64'}
Preview only (first 10 rows):
Capacity
     875
Full-file column statistics: {"Capacity": {"missing": 0, "unique_nonempty": 1, "numeric_range": [875.0, 875.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['ProductName', 'Weight', 'Value']
Parsed column types: {'ProductName': 'object', 'Weight': 'int64', 'Value': 'int64'}
Preview only (first 10 rows):
       ProductName Weight Value
           Spinach    230    64
Shiitake Mushrooms    637    75
            Apples    773    68
           Carrots    653    11
             Basil    755    91
          Potatoes    670    31
       Green Beans    505    90
       Blueberries    821    56
           Oranges     83    10
       Watermelons    249    24
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 10}, "Weight": {"missing": 0, "unique_nonempty": 10, "numeric_range": [83.0, 821.0]}, "Value": {"missing": 0, "unique_nonempty": 10, "numeric_range": [10.0, 91.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}]