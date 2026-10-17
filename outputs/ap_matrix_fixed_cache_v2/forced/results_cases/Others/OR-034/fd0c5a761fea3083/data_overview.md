File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 148
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
   Product Name Revenue Demand Initial Inventory
     12 MACARON      10    135               650
     ARMORICAIN     2.5      5                20
    ARTICLE 295       0      2                10
       BAGUETTE     0.9  38573            160220
 BAGUETTE APERO     4.5    128               660
BAGUETTE GRAINE     1.3   3390             15040
        BANETTE    1.05  38972            157590
      BANETTINE     0.6   5930             28230
   BOISSON 33CL     1.5   3376             14830
      BOTTEREAU     0.5    570              1500
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 148}, "Revenue": {"missing": 0, "unique_nonempty": 50, "numeric_range": [0.0, 35.0]}, "Demand": {"missing": 0, "unique_nonempty": 126, "numeric_range": [1.0, 193745.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 116, "numeric_range": [10.0, 725350.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}, {"term": "the", "matching_columns": [{"column": "Product Name", "exact": 1, "prefix": 1, "contains": 1, "examples": ["THE"]}], "exact_matching_columns": 1}]