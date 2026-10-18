File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 1
Columns: ['Capacity']
Parsed column types: {'Capacity': 'int64'}
Preview only (first 10 rows):
Capacity
     520
Full-file column statistics: {"Capacity": {"missing": 0, "unique_nonempty": 1, "numeric_range": [520.0, 520.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 20
Columns: ['ProductName', 'Value', 'Weight']
Parsed column types: {'ProductName': 'object', 'Value': 'int64', 'Weight': 'int64'}
Preview only (first 10 rows):
            ProductName Value Weight
                 NSAIDs   250    913
    Antirheumatic Drugs   178    754
Acetic Acid Derivatives   313    428
            Antibiotics   301    711
        Antiviral Drugs   425    350
      Antifungal Agents   260    159
        Antidepressants   848    353
         Antipsychotics   934    291
         Antihistamines   114    302
        Corticosteroids  1357     50
Full-file column statistics: {"ProductName": {"missing": 0, "unique_nonempty": 20}, "Value": {"missing": 0, "unique_nonempty": 16, "numeric_range": [114.0, 1780.0]}, "Weight": {"missing": 0, "unique_nonempty": 20, "numeric_range": [50.0, 913.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "products.csv", "matching_columns": [], "exact_matching_columns": 0}]