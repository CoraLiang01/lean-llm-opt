File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 7
Columns: ['Manager', 'Project 1 Cost', 'Project 2 Cost', 'Project 3 Cost', 'Project 4 Cost', 'Project 5 Cost', 'Project 6 Cost', 'Project 7 Cost']
Parsed column types: {'Manager': 'object', 'Project 1 Cost': 'int64', 'Project 2 Cost': 'int64', 'Project 3 Cost': 'int64', 'Project 4 Cost': 'int64', 'Project 5 Cost': 'int64', 'Project 6 Cost': 'int64', 'Project 7 Cost': 'int64'}
Preview only (first 10 rows):
  Manager Project 1 Cost Project 2 Cost Project 3 Cost Project 4 Cost Project 5 Cost Project 6 Cost Project 7 Cost
Manager 1           2972           2727           2795           2922           1302           2489           1533
Manager 2           1094           2158           2990           1844           2887           2021           2288
Manager 3           2133           1675           2422           2639           1033           2261           1695
Manager 4           1951           2309           2070           2802           2328           1313           2434
Manager 5           1269           2153           1296           2685           2627           1610           1641
Manager 6           1220           1192           2907           2622           2595           1261           2384
Manager 7           1286           1659           1179           1348           1420           2862           1959
Full-file column statistics: {"Manager": {"missing": 0, "unique_nonempty": 7}, "Project 1 Cost": {"missing": 0, "unique_nonempty": 7, "numeric_range": [1094.0, 2972.0]}, "Project 2 Cost": {"missing": 0, "unique_nonempty": 7, "numeric_range": [1192.0, 2727.0]}, "Project 3 Cost": {"missing": 0, "unique_nonempty": 7, "numeric_range": [1179.0, 2990.0]}, "Project 4 Cost": {"missing": 0, "unique_nonempty": 7, "numeric_range": [1348.0, 2922.0]}, "Project 5 Cost": {"missing": 0, "unique_nonempty": 7, "numeric_range": [1033.0, 2887.0]}, "Project 6 Cost": {"missing": 0, "unique_nonempty": 7, "numeric_range": [1261.0, 2862.0]}, "Project 7 Cost": {"missing": 0, "unique_nonempty": 7, "numeric_range": [1533.0, 2434.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "manager", "matching_columns": [{"column": "Manager", "exact": 0, "prefix": 7, "contains": 7, "examples": ["Manager 1", "Manager 2", "Manager 3"]}], "exact_matching_columns": 0}, {"term": "manager_project_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]