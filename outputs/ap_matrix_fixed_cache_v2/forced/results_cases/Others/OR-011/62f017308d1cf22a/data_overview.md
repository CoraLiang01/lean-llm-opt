File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 900
Columns: ['id_number', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'id_number': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
id_number Revenue Demand Initial Inventory
    id100  127.57   7857             53990
    id101  245.08   8626             59620
    id102   341.4   7984             55100
    id103   127.6   7881             54510
    id104   31.19   8297             57510
    id105  167.45   8118             56270
    id106  399.92   8422             58230
    id107   171.3   7881             54550
    id108  412.76   8078             55960
    id109   66.54   7756             53640
Full-file column statistics: {"id_number": {"missing": 0, "unique_nonempty": 900}, "Revenue": {"missing": 0, "unique_nonempty": 892, "numeric_range": [6.14, 499.93]}, "Demand": {"missing": 0, "unique_nonempty": 588, "numeric_range": [7180.0, 8865.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 494, "numeric_range": [49500.0, 61440.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "id999", "matching_columns": [{"column": "id_number", "exact": 1, "prefix": 1, "contains": 1, "examples": ["id999"]}], "exact_matching_columns": 1}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]