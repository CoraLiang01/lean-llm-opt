File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 1
Columns: ['Capacity']
Parsed column types: {'Capacity': 'int64'}
Preview only (first 10 rows):
Capacity
    1035
Full-file column statistics: {"Capacity": {"missing": 0, "unique_nonempty": 1, "numeric_range": [1035.0, 1035.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['ProductName', 'Weight', 'Value']
Parsed column types: {'ProductName': 'object', 'Weight': 'int64', 'Value': 'int64'}
Preview only (first 10 rows):
       ProductName Weight Value
           Spinach    282    49
Shiitake Mushrooms     83    30
            Apples    251    30
           Carrots    257    18
             Basil     88    54
          Potatoes     52    27
       Green Beans    198    91
       Blueberries    203    88
           Oranges     87    78
       Watermelons    265    22
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 10}, "Weight": {"missing": 0, "unique_nonempty": 10, "numeric_range": [52.0, 282.0]}, "Value": {"missing": 0, "unique_nonempty": 9, "numeric_range": [18.0, 91.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [], "exact_matching_columns": 0}]