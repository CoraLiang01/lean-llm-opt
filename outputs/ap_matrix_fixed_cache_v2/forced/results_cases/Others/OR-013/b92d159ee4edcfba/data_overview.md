File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM4/OnlineSalesinUSA.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 47932
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
                   Product Name Revenue Demand Initial Inventory
           jjp_15000006-100-NIL   328.5      3                20
                  4U_Service_22    56.0      5                30
                  4U_Service_36    21.6      3                20
                   4U_Service_7    62.5      3                20
               7CF5AFBC3B16A32A    35.9     14               110
A4-Tech_Head-Phone-M-660-Bloody   444.4      2                10
               ABR5AE2FBE29498D    85.0      6                40
 ABT_AUV140-32G-RBE-32GB-UV-140   139.9      3                20
                      ABT_G100s   349.9      3                20
                       ABT_K230   159.9      3                20
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 47932}, "Revenue": {"missing": 0, "unique_nonempty": 6403, "numeric_range": [0.0, 101262.59]}, "Demand": {"missing": 0, "unique_nonempty": 755, "numeric_range": [2.0, 13473.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 605, "numeric_range": [10.0, 97620.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "4u", "matching_columns": [{"column": "Product Name", "exact": 0, "prefix": 3, "contains": 18, "examples": ["4U_Service_22", "4U_Service_36", "4U_Service_7"]}], "exact_matching_columns": 0}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]