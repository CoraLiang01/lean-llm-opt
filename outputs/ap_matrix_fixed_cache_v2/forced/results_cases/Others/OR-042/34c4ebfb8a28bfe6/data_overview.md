File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 1
Columns: ['Capacity']
Parsed column types: {'Capacity': 'int64'}
Preview only (first 10 rows):
Capacity
    4120
Full-file column statistics: {"Capacity": {"missing": 0, "unique_nonempty": 1, "numeric_range": [4120.0, 4120.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 20
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'object', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
            ProductName Value Weight
                 NSAIDs   585     50
    Antirheumatic Drugs   557    329
Acetic Acid Derivatives   963    410
            Antibiotics   301    452
        Antiviral Drugs   425    350
      Antifungal Agents   260    159
        Antidepressants   848    353
         Antipsychotics   461    291
         Antihistamines   840    302
        Corticosteroids   999     50
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 20}, "Value": {"missing": 0, "unique_nonempty": 20, "numeric_range": [260.0, 999.0]}, "Weight": {"missing": 0, "unique_nonempty": 19, "numeric_range": [50.0, 470.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "antirheumatic drugs", "matching_columns": [{"column": "ProductName", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Antirheumatic Drugs"]}], "exact_matching_columns": 1}, {"term": "capacity.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "nsaids", "matching_columns": [{"column": "ProductName", "exact": 1, "prefix": 1, "contains": 1, "examples": ["NSAIDs"]}], "exact_matching_columns": 1}, {"term": "products.csv.", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [], "exact_matching_columns": 0}]