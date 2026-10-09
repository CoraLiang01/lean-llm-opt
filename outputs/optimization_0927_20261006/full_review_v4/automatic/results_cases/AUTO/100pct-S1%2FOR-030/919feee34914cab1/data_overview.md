File: /Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 110
Columns: ['Project ID', 'ArchiveAccessRoute', 'ArchivePageCount', 'ArchiveRevisionCount', 'ArchiveFolder', 'Project Name', 'Capital (k$)', 'NPV (k$)']
Parsed column types: {'Project ID': 'int64', 'ArchiveAccessRoute': 'object', 'ArchivePageCount': 'int64', 'ArchiveRevisionCount': 'int64', 'ArchiveFolder': 'object', 'Project Name': 'object', 'Capital (k$)': 'int64', 'NPV (k$)': 'int64'}
Preview only (first 10 rows):
Project ID ArchiveAccessRoute ArchivePageCount ArchiveRevisionCount ArchiveFolder                 Project Name Capital (k$) NPV (k$)
         1         LocalIndex                6                    1      Folder_A       Infrastructure Upgrade           50       60
         2         LocalIndex               20                   19      Folder_C           New Product Line A           40       50
         3         LocalIndex                2                    1      Folder_B         Marketing Campaign X           30       45
         4       CatalogIndex               16                   23      Folder_C         R&D Initiative Alpha           25       35
         5             Portal                7                    6      Folder_B       Staff Training Program           20       28
         6         LocalIndex               21                    2      Folder_C            System Automation           65       75
         7         LocalIndex               17                    6      Folder_C       Global Expansion Pilot           80      100
         8       CatalogIndex               27                   14      Folder_B          Green Energy Switch           15       20
         9             Portal               18                   19      Folder_C       Warehouse Optimization           48       62
        10         LocalIndex               19                   30      Folder_C Customer Experience Platform           55       70
Full-file column statistics: {"Project ID": {"missing": 0, "unique_nonempty": 110, "numeric_range": [1.0, 110.0]}, "ArchiveAccessRoute": {"missing": 0, "unique_nonempty": 3}, "ArchivePageCount": {"missing": 0, "unique_nonempty": 30, "numeric_range": [1.0, 30.0]}, "ArchiveRevisionCount": {"missing": 0, "unique_nonempty": 29, "numeric_range": [1.0, 30.0]}, "ArchiveFolder": {"missing": 0, "unique_nonempty": 3}, "Project Name": {"missing": 0, "unique_nonempty": 110}, "Capital (k$)": {"missing": 0, "unique_nonempty": 73, "numeric_range": [11.0, 149.0]}, "NPV (k$)": {"missing": 0, "unique_nonempty": 86, "numeric_range": [14.0, 205.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer experience platform", "matching_columns": [{"column": "Project Name", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Customer Experience Platform"]}], "exact_matching_columns": 1}, {"term": "global expansion pilot", "matching_columns": [{"column": "Project Name", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Global Expansion Pilot"]}], "exact_matching_columns": 1}, {"term": "infrastructure upgrade", "matching_columns": [{"column": "Project Name", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Infrastructure Upgrade"]}], "exact_matching_columns": 1}, {"term": "r&d initiative alpha", "matching_columns": [{"column": "Project Name", "exact": 1, "prefix": 1, "contains": 1, "examples": ["R&D Initiative Alpha"]}], "exact_matching_columns": 1}, {"term": "system automation", "matching_columns": [{"column": "Project Name", "exact": 1, "prefix": 1, "contains": 1, "examples": ["System Automation"]}], "exact_matching_columns": 1}]