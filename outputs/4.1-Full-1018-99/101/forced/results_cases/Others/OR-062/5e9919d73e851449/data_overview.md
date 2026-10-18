File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5
Columns: ['Customer', 'demand']
Parsed column types: {'Customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
  Customer demand
Customer_1   2397
Customer_2   1889
Customer_3   2518
Customer_4   3218
Customer_5   1813
Full-file column statistics: {"Customer": {"missing": 0, "unique_nonempty": 5}, "demand": {"missing": 0, "unique_nonempty": 5, "numeric_range": [1813.0, 3218.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "e", "matching_columns": [{"column": "Customer", "exact": 0, "prefix": 0, "contains": 5, "examples": ["Customer_1", "Customer_2", "Customer_3"]}], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5
Columns: ['Unnamed: 0', 'fixed_costs']
Parsed column types: {'Unnamed: 0': 'object', 'fixed_costs': 'float64'}
Preview only (first 10 rows):
Unnamed: 0       fixed_costs
 MOUNT AYR             96.58
    WAUKEE             94.06
   WAVERLY             94.37
     PELLA             82.88
DES MOINES 94.95999999999999
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 5}, "fixed_costs": {"missing": 0, "unique_nonempty": 5, "numeric_range": [82.88, 96.58]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "e", "matching_columns": [{"column": "Unnamed: 0", "exact": 0, "prefix": 0, "contains": 4, "examples": ["WAUKEE", "WAVERLY", "PELLA"]}], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5
Columns: ['Unnamed: 0', 'CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT']
Parsed column types: {'Unnamed: 0': 'object', 'CLARINDA': 'float64', 'FORT MADISON': 'float64', 'SIOUX CITY': 'float64', 'TOLEDO': 'float64', 'BANCROFT': 'float64'}
Preview only (first 10 rows):
Unnamed: 0          CLARINDA FORT MADISON        SIOUX CITY  TOLEDO BANCROFT
 MOUNT AYR 694.6799999999999        17.48             20.07  199.02  1685.53
    WAUKEE             15.13          1.5              1.43   27.88    90.69
   WAVERLY              2.34       349.34             246.6    41.3    78.73
     PELLA            1181.6      1458.53           1646.36 1924.55    38.93
DES MOINES            1030.8        43.48 932.4299999999999   55.39   103.84
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 5}, "CLARINDA": {"missing": 0, "unique_nonempty": 5, "numeric_range": [2.34, 1181.6]}, "FORT MADISON": {"missing": 0, "unique_nonempty": 5, "numeric_range": [1.5, 1458.53]}, "SIOUX CITY": {"missing": 0, "unique_nonempty": 5, "numeric_range": [1.43, 1646.36]}, "TOLEDO": {"missing": 0, "unique_nonempty": 5, "numeric_range": [27.88, 1924.55]}, "BANCROFT": {"missing": 0, "unique_nonempty": 5, "numeric_range": [38.93, 1685.53]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "e", "matching_columns": [{"column": "Unnamed: 0", "exact": 0, "prefix": 0, "contains": 4, "examples": ["WAUKEE", "WAVERLY", "PELLA"]}], "exact_matching_columns": 0}, {"term": "fixed_cost.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]