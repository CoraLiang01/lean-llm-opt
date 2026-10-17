File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 1
Columns: ['Capacity']
Parsed column types: {'Capacity': 'int64'}
Preview only (first 10 rows):
Capacity
     586
Full-file column statistics: {"Capacity": {"missing": 0, "unique_nonempty": 1, "numeric_range": [586.0, 586.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 20
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'object', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
     ProductName Value Weight
          Queens   469    954
        Brooklyn   290    650
       Manhattan   236    961
           Bronx   235    950
   Staten Island   745    379
          Harlem   684    776
 Upper East Side   444    381
 Lower Manhattan   172    808
         Midtown  1000    937
Long Island City   336    608
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 20}, "Value": {"missing": 0, "unique_nonempty": 20, "numeric_range": [139.0, 1000.0]}, "Weight": {"missing": 0, "unique_nonempty": 20, "numeric_range": [130.0, 961.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "brooklyn", "matching_columns": [{"column": "ProductName", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Brooklyn"]}], "exact_matching_columns": 1}, {"term": "capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "queens", "matching_columns": [{"column": "ProductName", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Queens"]}], "exact_matching_columns": 1}]