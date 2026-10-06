File: /Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S2/Large-scale-or/Others_example/Others3/value.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 140
Columns: ['item', 'ItemCatalogChannel', 'value', 'ItemSupportInquiryCountLastQuarter', 'DisplayInquiryCountLastMonth', 'weight', 'ItemDocumentationTeam', 'ItemPhotoCount', 'MerchandisingTeam']
Parsed column types: {'item': 'int64', 'ItemCatalogChannel': 'object', 'value': 'int64', 'ItemSupportInquiryCountLastQuarter': 'int64', 'DisplayInquiryCountLastMonth': 'int64', 'weight': 'int64', 'ItemDocumentationTeam': 'object', 'ItemPhotoCount': 'int64', 'MerchandisingTeam': 'object'}
Preview only (first 10 rows):
item ItemCatalogChannel value ItemSupportInquiryCountLastQuarter DisplayInquiryCountLastMonth weight ItemDocumentationTeam ItemPhotoCount MerchandisingTeam
   1         WebCatalog    10                                 28                           25      2                Team_B             10            Team_C
   2         WebCatalog    40                                 10                           23      5                Team_A             11            Team_C
   3       PrintCatalog    30                                  2                           29      4                Team_B             26            Team_C
   4         WebCatalog    50                                  6                            4      8                Team_B             11            Team_A
   5       StoreCatalog    35                                 11                            5      7                Team_A             22            Team_B
   6         WebCatalog     6                                 20                           21      1                Team_A             16            Team_A
   7         WebCatalog    33                                 20                            8      8                Team_C             22            Team_C
   8       StoreCatalog    57                                 15                           12      7                Team_C             23            Team_C
   9         WebCatalog    42                                 17                           24      5                Team_A             11            Team_A
  10         WebCatalog    24                                 19                            7      5                Team_B              9            Team_A
Full-file column statistics: {"item": {"missing": 0, "unique_nonempty": 140, "numeric_range": [1.0, 140.0]}, "ItemCatalogChannel": {"missing": 0, "unique_nonempty": 3}, "value": {"missing": 0, "unique_nonempty": 62, "numeric_range": [4.0, 82.0]}, "ItemSupportInquiryCountLastQuarter": {"missing": 0, "unique_nonempty": 30, "numeric_range": [1.0, 30.0]}, "DisplayInquiryCountLastMonth": {"missing": 0, "unique_nonempty": 30, "numeric_range": [1.0, 30.0]}, "weight": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1.0, 10.0]}, "ItemDocumentationTeam": {"missing": 0, "unique_nonempty": 3}, "ItemPhotoCount": {"missing": 0, "unique_nonempty": 30, "numeric_range": [1.0, 30.0]}, "MerchandisingTeam": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "value", "matching_columns": [], "exact_matching_columns": 0}, {"term": "weight", "matching_columns": [], "exact_matching_columns": 0}]