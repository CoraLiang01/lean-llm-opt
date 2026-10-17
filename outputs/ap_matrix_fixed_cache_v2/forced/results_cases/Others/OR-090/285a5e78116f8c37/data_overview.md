File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 100
Columns: ['product', 'profit_per_unit', 'r1_per_unit', 'r2_per_unit', 'r3_per_unit', 'upper_demand_units', 'batch_size_units']
Parsed column types: {'product': 'object', 'profit_per_unit': 'float64', 'r1_per_unit': 'float64', 'r2_per_unit': 'float64', 'r3_per_unit': 'float64', 'upper_demand_units': 'int64', 'batch_size_units': 'int64'}
Preview only (first 10 rows):
product profit_per_unit r1_per_unit r2_per_unit r3_per_unit upper_demand_units batch_size_units
     P1             6.7        2.37        0.61        2.03                317               10
     P2           10.96        4.79        2.73        0.53                106               10
     P3             9.4        3.87         1.6        0.74                386               10
     P4           11.13        3.31        2.28        2.73                441               10
     P5           10.29        1.46        3.68        1.94                 63               10
     P6            8.06        1.46        1.37        0.32                221               10
     P7            6.94        1.04        1.94        0.57                441               10
     P8           11.44        4.44        3.14        2.09                489               10
     P9            9.13        3.32         1.3        0.31                277               10
    P10            7.83        3.77        0.77        0.73                121               10
Full-file column statistics: {"product": {"missing": 0, "unique_nonempty": 100}, "profit_per_unit": {"missing": 0, "unique_nonempty": 93, "numeric_range": [5.49, 12.73]}, "r1_per_unit": {"missing": 0, "unique_nonempty": 85, "numeric_range": [0.82, 4.94]}, "r2_per_unit": {"missing": 0, "unique_nonempty": 88, "numeric_range": [0.52, 3.95]}, "r3_per_unit": {"missing": 0, "unique_nonempty": 88, "numeric_range": [0.31, 2.97]}, "upper_demand_units": {"missing": 0, "unique_nonempty": 92, "numeric_range": [45.0, 613.0]}, "batch_size_units": {"missing": 0, "unique_nonempty": 1, "numeric_range": [10.0, 10.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "batch_size_units", "matching_columns": [], "exact_matching_columns": 0}, {"term": "p1", "matching_columns": [{"column": "product", "exact": 1, "prefix": 12, "contains": 12, "examples": ["P1", "P10", "P11"]}], "exact_matching_columns": 1}, {"term": "p100", "matching_columns": [{"column": "product", "exact": 1, "prefix": 1, "contains": 1, "examples": ["P100"]}], "exact_matching_columns": 1}, {"term": "product", "matching_columns": [], "exact_matching_columns": 0}, {"term": "r1_per_unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "r2_per_unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "r3_per_unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "upper_demand_units", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 3
Columns: ['resource', 'capacity']
Parsed column types: {'resource': 'object', 'capacity': 'float64'}
Preview only (first 10 rows):
resource capacity
      R1 27380.54
      R2 22245.11
      R3 15147.73
Full-file column statistics: {"resource": {"missing": 0, "unique_nonempty": 3}, "capacity": {"missing": 0, "unique_nonempty": 3, "numeric_range": [15147.73, 27380.54]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "r1", "matching_columns": [{"column": "resource", "exact": 1, "prefix": 1, "contains": 1, "examples": ["R1"]}], "exact_matching_columns": 1}, {"term": "r2", "matching_columns": [{"column": "resource", "exact": 1, "prefix": 1, "contains": 1, "examples": ["R2"]}], "exact_matching_columns": 1}, {"term": "r3", "matching_columns": [{"column": "resource", "exact": 1, "prefix": 1, "contains": 1, "examples": ["R3"]}], "exact_matching_columns": 1}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}]