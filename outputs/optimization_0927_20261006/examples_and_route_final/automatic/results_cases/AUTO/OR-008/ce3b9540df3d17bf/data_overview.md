File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['Customers', 'demand']
Parsed column types: {'Customers': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
Customers demand
Customer1     70
Customer2     80
Customer3     60
Customer4     90
Customer5     85
Customer6     95
Full-file column statistics: {"Customers": {"missing": 0, "unique_nonempty": 6}, "demand": {"missing": 0, "unique_nonempty": 6, "numeric_range": [60.0, 95.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "freshmart,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5
Columns: ['Suppliers', 'supply_capacity']
Parsed column types: {'Suppliers': 'object', 'supply_capacity': 'int64'}
Preview only (first 10 rows):
Suppliers supply_capacity
Supplier1             200
Supplier2             250
Supplier3             230
Supplier4             220
Supplier5             210
Full-file column statistics: {"Suppliers": {"missing": 0, "unique_nonempty": 5}, "supply_capacity": {"missing": 0, "unique_nonempty": 5, "numeric_range": [200.0, 250.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "freshmart,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5
Columns: ['Unnamed: 0', 'Customer1', 'Customer2', 'Customer3', 'Customer4', 'Customer5', 'Customer6']
Parsed column types: {'Unnamed: 0': 'object', 'Customer1': 'int64', 'Customer2': 'int64', 'Customer3': 'int64', 'Customer4': 'int64', 'Customer5': 'int64', 'Customer6': 'int64'}
Preview only (first 10 rows):
Unnamed: 0 Customer1 Customer2 Customer3 Customer4 Customer5 Customer6
 Supplier1         2         3         1         2         3         2
 Supplier2         1         2         3         2         3         2
 Supplier3         3         1         2         3         2         3
 Supplier4         2         3         2         1         3         4
 Supplier5         3         2         3         3         2         3
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 5}, "Customer1": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "Customer2": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "Customer3": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "Customer4": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "Customer5": {"missing": 0, "unique_nonempty": 2, "numeric_range": [2.0, 3.0]}, "Customer6": {"missing": 0, "unique_nonempty": 3, "numeric_range": [2.0, 4.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "freshmart,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]