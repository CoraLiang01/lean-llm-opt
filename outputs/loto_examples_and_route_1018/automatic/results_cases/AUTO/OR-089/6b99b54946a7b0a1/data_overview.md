File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 15
Columns: ['Customer', 'SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6', 'SC7', 'SC8', 'SC9', 'SC10']
Parsed column types: {'Customer': 'object', 'SC1': 'float64', 'SC2': 'float64', 'SC3': 'float64', 'SC4': 'float64', 'SC5': 'float64', 'SC6': 'float64', 'SC7': 'float64', 'SC8': 'float64', 'SC9': 'float64', 'SC10': 'float64'}
Preview only (first 10 rows):
Customer  SC1  SC2  SC3  SC4  SC5  SC6  SC7  SC8  SC9 SC10
      C1 15.1 21.2 14.9 18.8 22.9 16.8 16.5  9.4 16.1 17.3
      C2 13.4 16.3 20.2 19.6 20.9 22.1 16.9  9.4 13.8 11.7
      C3 15.2 18.8 14.7 21.7 18.1 18.6 12.3 11.2 11.9 20.4
      C4 16.8 19.1 18.3 18.8 23.1 15.7 13.1  8.6 15.6 22.2
      C5 13.4 18.6 20.8 19.8 22.1 18.1 16.7 12.1 11.4 18.2
      C6 12.5 22.5 15.5 14.9 21.6 21.3 16.1 10.7 11.9 14.6
      C7 12.1 17.1 19.8 18.6 22.1 20.7 20.5 12.2 15.4 18.7
      C8 12.3 15.7 17.9 21.3 22.7 15.3 16.6 11.4 14.1 20.1
      C9 16.3 21.3 17.6 20.8 21.8 17.2 15.5 12.6 19.9 19.1
     C10 12.1 18.7 14.4 20.1 22.7 14.1 18.1 11.4 18.1 17.4
Full-file column statistics: {"Customer": {"missing": 0, "unique_nonempty": 15}, "SC1": {"missing": 0, "unique_nonempty": 11, "numeric_range": [8.3, 16.8]}, "SC2": {"missing": 0, "unique_nonempty": 13, "numeric_range": [15.7, 23.8]}, "SC3": {"missing": 0, "unique_nonempty": 13, "numeric_range": [14.4, 20.8]}, "SC4": {"missing": 0, "unique_nonempty": 14, "numeric_range": [14.9, 21.7]}, "SC5": {"missing": 0, "unique_nonempty": 12, "numeric_range": [18.1, 24.2]}, "SC6": {"missing": 0, "unique_nonempty": 15, "numeric_range": [14.1, 22.1]}, "SC7": {"missing": 0, "unique_nonempty": 14, "numeric_range": [12.3, 20.5]}, "SC8": {"missing": 0, "unique_nonempty": 13, "numeric_range": [8.5, 16.7]}, "SC9": {"missing": 0, "unique_nonempty": 13, "numeric_range": [11.1, 19.9]}, "SC10": {"missing": 0, "unique_nonempty": 14, "numeric_range": [11.7, 22.2]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "c1", "matching_columns": [{"column": "Customer", "exact": 1, "prefix": 7, "contains": 7, "examples": ["C1", "C10", "C11"]}], "exact_matching_columns": 1}, {"term": "c15", "matching_columns": [{"column": "Customer", "exact": 1, "prefix": 1, "contains": 1, "examples": ["C15"]}], "exact_matching_columns": 1}, {"term": "customer", "matching_columns": [], "exact_matching_columns": 0}, {"term": "sc1", "matching_columns": [], "exact_matching_columns": 0}, {"term": "sc10", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['Service Center', 'Fixed Opening Cost']
Parsed column types: {'Service Center': 'object', 'Fixed Opening Cost': 'float64'}
Preview only (first 10 rows):
Service Center Fixed Opening Cost
           SC1              385.1
           SC2              546.3
           SC3              485.2
           SC4              448.1
           SC5              324.1
           SC6              323.9
           SC7              296.5
           SC8              522.7
           SC9              448.7
          SC10              478.7
Full-file column statistics: {"Service Center": {"missing": 0, "unique_nonempty": 10}, "Fixed Opening Cost": {"missing": 0, "unique_nonempty": 10, "numeric_range": [296.5, 546.3]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fixed opening cost", "matching_columns": [], "exact_matching_columns": 0}, {"term": "sc1", "matching_columns": [{"column": "Service Center", "exact": 1, "prefix": 2, "contains": 2, "examples": ["SC1", "SC10"]}], "exact_matching_columns": 1}, {"term": "sc10", "matching_columns": [{"column": "Service Center", "exact": 1, "prefix": 1, "contains": 1, "examples": ["SC10"]}], "exact_matching_columns": 1}]