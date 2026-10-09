File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 194
Columns: ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
Parsed column types: {'Product Name': 'object', 'Revenue': 'float64', 'Demand': 'int64', 'Initial Inventory': 'int64'}
Preview only (first 10 rows):
                              Product Name Revenue Demand Initial Inventory
           100% FEATHER FILL PUFFER JACKET   169.0   4030             30030
                      100% LINEN OVERSHIRT    89.9   3348             24740
                     100% WOOL SUIT JACKET   169.0   2609             19160
                 ABSTRACT JACQUARD SWEATER    59.9    736              5290
               ABSTRACT PRINT KNIT T-SHIRT    39.9   2777             20930
                    ABSTRACT PRINT T-SHIRT    39.9   1238              9960
                    ACID WASH DENIM JACKET    89.9   2683             19170
                 ADHERENT STRIPES SNEAKERS    45.9   1369              9910
ALPACA AND WOOL BLEND TIE DYE KNIT SWEATER    49.9   1367              9940
            ALPACA BLEND OPEN KNIT SWEATER    79.9   2455             17360
Full-file column statistics: {"Product Name": {"missing": 0, "unique_nonempty": 194}, "Revenue": {"missing": 0, "unique_nonempty": 28, "numeric_range": [7.99, 439.0]}, "Demand": {"missing": 0, "unique_nonempty": 190, "numeric_range": [658.0, 14546.0]}, "Initial Inventory": {"missing": 0, "unique_nonempty": 188, "numeric_range": [5290.0, 109100.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "faux", "matching_columns": [{"column": "Product Name", "exact": 0, "prefix": 12, "contains": 12, "examples": ["FAUX FUR JEWEL SWEATER", "FAUX LEATHER BOMBER JACKET", "FAUX LEATHER BOXY FIT JACKET"]}], "exact_matching_columns": 0}, {"term": "initial inventory", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revenue", "matching_columns": [], "exact_matching_columns": 0}]