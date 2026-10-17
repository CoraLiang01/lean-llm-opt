File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 3
Columns: ['Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)']
Parsed column types: {'Grade': 'object', 'Daily Supply (kg)': 'int64', 'Cost (CNY/kg)': 'float64'}
Preview only (first 10 rows):
Grade Daily Supply (kg) Cost (CNY/kg)
    I              1500             6
   II              2000           4.5
  III              1000             3
Full-file column statistics: {"Grade": {"missing": 0, "unique_nonempty": 3}, "Daily Supply (kg)": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1000.0, 2000.0]}, "Cost (CNY/kg)": {"missing": 0, "unique_nonempty": 3, "numeric_range": [3.0, 6.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "grade", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 3
Columns: ['Brand', 'Blending Requirements', 'Selling Price (CNY/kg)']
Parsed column types: {'Brand': 'object', 'Blending Requirements': 'object', 'Selling Price (CNY/kg)': 'float64'}
Preview only (first 10 rows):
 Brand              Blending Requirements Selling Price (CNY/kg)
   Red  I less than 10%  II more than 50%                    5.5
Yellow III less than 70%  I more than 20%                      5
  Blue  I less than 50%  II more than 10%                    4.8
Full-file column statistics: {"Brand": {"missing": 0, "unique_nonempty": 3}, "Blending Requirements": {"missing": 0, "unique_nonempty": 3}, "Selling Price (CNY/kg)": {"missing": 0, "unique_nonempty": 3, "numeric_range": [4.8, 5.5]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "blending requirements", "matching_columns": [], "exact_matching_columns": 0}, {"term": "brand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "red", "matching_columns": [{"column": "Brand", "exact": 1, "prefix": 1, "contains": 1, "examples": ["Red"]}], "exact_matching_columns": 1}]