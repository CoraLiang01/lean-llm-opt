File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5
Columns: ['customer', 'demand']
Parsed column types: {'customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
customer demand
      D1    428
      D2    217
      D3    214
      D4    380
      D5    254
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 5}, "demand": {"missing": 0, "unique_nonempty": 5, "numeric_range": [214.0, 428.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "greenmart,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5
Columns: ['region', 'supply_capacity']
Parsed column types: {'region': 'object', 'supply_capacity': 'int64'}
Preview only (first 10 rows):
region supply_capacity
    S1             428
    S2             217
    S3             214
    S4             380
    S5             254
Full-file column statistics: {"region": {"missing": 0, "unique_nonempty": 5}, "supply_capacity": {"missing": 0, "unique_nonempty": 5, "numeric_range": [214.0, 428.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "greenmart,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5
Columns: ['Unnamed: 0', 'D1', 'D2', 'D3', 'D4', 'D5']
Parsed column types: {'Unnamed: 0': 'object', 'D1': 'float64', 'D2': 'float64', 'D3': 'float64', 'D4': 'float64', 'D5': 'float64'}
Preview only (first 10 rows):
Unnamed: 0                 D1                 D2                 D3                 D4                 D5
        S1  269.3910588020795 1.4537335390933939  99.60345345756605  26.64078166309837  9.537688956880922
        S2  9.291846876785183 10.874778437070223 144.52609291614627 11.420133077898234  153.1756819927813
        S3  9.674584301671008 2.6191650959687944  100.8242249168735 3.2121910887916876  133.8493396124168
        S4 270.57498480010247        32.50253586 4.6842098096469815 1.5682269686546804         9.58927599
        S5  226.0331910675782  8.669161980826471  65.47681316968448  9.068765258459958 202.65015316425533
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 5}, "D1": {"missing": 0, "unique_nonempty": 5, "numeric_range": [9.291846876785185, 270.57498480010247]}, "D2": {"missing": 0, "unique_nonempty": 5, "numeric_range": [1.453733539093394, 32.50253586]}, "D3": {"missing": 0, "unique_nonempty": 5, "numeric_range": [4.6842098096469815, 144.52609291614627]}, "D4": {"missing": 0, "unique_nonempty": 5, "numeric_range": [1.5682269686546804, 26.64078166309837]}, "D5": {"missing": 0, "unique_nonempty": 5, "numeric_range": [9.537688956880922, 202.65015316425533]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "greenmart,", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv.", "matching_columns": [], "exact_matching_columns": 0}]