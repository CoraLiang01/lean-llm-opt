File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_01/export_01.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 50
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'component', 'amount', 'currency']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'component': 'object', 'amount': 'int64', 'currency': 'object'}
Preview only (first 10 rows):
  table dealership_id    record_id revision effective_date action      item_ref     component amount currency
benefit OSLO_NEW_CARS benefit:0049        2     2026-04-30 UPSERT R45289ace3322 item24_rebate    -28      EUR
benefit OSLO_NEW_CARS benefit:0028        2     2026-04-30 UPSERT Rde7df7983569   item14_base   2882      EUR
benefit OSLO_NEW_CARS benefit:0031        1     2026-04-02 UPSERT Rbbac12e00a0c item15_rebate  -3793      JPY
benefit OSLO_NEW_CARS benefit:0022        1     2026-04-02 UPSERT Rf9d792d5328c   item11_base   1416      USD
benefit OSLO_NEW_CARS benefit:0019        2     2026-04-30 UPSERT R826527278104  item9_rebate  -1300      JPY
benefit OSLO_NEW_CARS benefit:0046        1     2026-04-02 UPSERT R5177e1b63b51   item23_base  33507      JPY
benefit OSLO_NEW_CARS benefit:0010        1     2026-04-02 UPSERT Rb30197d6899f    item5_base   1113      USD
benefit OSLO_NEW_CARS benefit:0004        2     2026-04-30 UPSERT R2dd6dc93369f    item2_base   1508      USD
benefit OSLO_NEW_CARS benefit:0037        2     2026-04-30 UPSERT R8efb61b6c33b item18_rebate    -74      EUR
benefit OSLO_NEW_CARS benefit:0055        2     2026-04-30 UPSERT R3127c4fdc4b8 item27_rebate  -3500      JPY
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 24}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 19}, "component": {"missing": 0, "unique_nonempty": 24}, "amount": {"missing": 0, "unique_nonempty": 41, "numeric_range": [-3800.0, 126500.0]}, "currency": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "benefit", "matching_columns": [{"column": "table", "exact": 50, "prefix": 50, "contains": 50, "examples": ["benefit"]}, {"column": "record_id", "exact": 0, "prefix": 50, "contains": 50, "examples": ["benefit:0049", "benefit:0028", "benefit:0031"]}], "exact_matching_columns": 1}, {"term": "currency", "matching_columns": [], "exact_matching_columns": 0}, {"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 45, "prefix": 45, "contains": 45, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usd", "matching_columns": [{"column": "currency", "exact": 24, "prefix": 24, "contains": 24, "examples": ["USD"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_02/export_02.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 49
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'component', 'amount', 'currency']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'component': 'object', 'amount': 'float64', 'currency': 'object'}
Preview only (first 10 rows):
  table dealership_id         record_id revision effective_date action      item_ref     component amount currency
benefit OSLO_NEW_CARS      benefit:0023        2     2026-04-30 UPSERT Rf9d792d5328c item11_rebate    -46      EUR
benefit OSLO_NEW_CARS      benefit:0026        2     2026-04-30 UPSERT Re4cddba9a9d6   item13_base  82000      JPY
benefit OSLO_NEW_CARS      benefit:0044        1     2026-04-02 UPSERT R30a2c98896e0   item22_base   1093      USD
benefit OSLO_NEW_CARS      benefit:0014        2     2026-04-30 UPSERT Rc6cd5cbe3d08    item7_base    872      EUR
benefit OSLO_NEW_CARS      benefit:0026        1     2026-04-02 UPSERT Re4cddba9a9d6   item13_base  82007      JPY
benefit OSLO_NEW_CARS      benefit:0032        2     2026-04-30 UPSERT R17fb1a9043c3   item16_base   2522      EUR
benefit OSLO_NEW_CARS      benefit:0002        1     2026-04-02 UPSERT Rb981c8b74356    item1_base    801      EUR
benefit OSLO_NEW_CARS      benefit:0014        1     2026-04-02 UPSERT Rc6cd5cbe3d08    item7_base    879      EUR
benefit OSLO_NEW_CARS benefit:withdrawn        1     2026-04-02 UPSERT Re43073e4a0c5    item0_base   1017      USD
benefit OSLO_NEW_CARS benefit:withdrawn        2     2026-05-05 DELETE                                            
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 24}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_ref": {"missing": 1, "unique_nonempty": 19}, "component": {"missing": 1, "unique_nonempty": 24}, "amount": {"missing": 1, "unique_nonempty": 42, "numeric_range": [-3800.0, 901007.0]}, "currency": {"missing": 1, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "benefit", "matching_columns": [{"column": "table", "exact": 49, "prefix": 49, "contains": 49, "examples": ["benefit"]}, {"column": "record_id", "exact": 0, "prefix": 49, "contains": 49, "examples": ["benefit:0023", "benefit:0026", "benefit:0044"]}], "exact_matching_columns": 1}, {"term": "currency", "matching_columns": [], "exact_matching_columns": 0}, {"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 44, "prefix": 44, "contains": 44, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usd", "matching_columns": [{"column": "currency", "exact": 20, "prefix": 20, "contains": 20, "examples": ["USD"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_03/export_03.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 49
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'component', 'amount', 'currency']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'component': 'object', 'amount': 'int64', 'currency': 'object'}
Preview only (first 10 rows):
  table   dealership_id    record_id revision effective_date action      item_ref     component amount currency
benefit   OSLO_NEW_CARS benefit:0039        1     2026-04-02 UPSERT R777716c59761 item19_rebate     -3      USD
benefit   OSLO_NEW_CARS benefit:0018        2     2026-04-30 UPSERT R826527278104    item9_base  60000      JPY
benefit   OSLO_NEW_CARS benefit:0009        1     2026-04-02 UPSERT R6121e8a8833d  item4_rebate    -28      USD
benefit   OSLO_NEW_CARS benefit:0015        3     2026-05-19 UPSERT Rc6cd5cbe3d08  item7_rebate    -61      EUR
benefit   OSLO_NEW_CARS benefit:0033        1     2026-04-02 UPSERT R17fb1a9043c3 item16_rebate    -21      USD
benefit   OSLO_NEW_CARS benefit:0045        1     2026-04-02 UPSERT R30a2c98896e0 item22_rebate    -15      USD
benefit BERGEN_NEW_CARS benefit:0008        2     2026-04-30 UPSERT R6121e8a8833d    item4_base    489      USD
benefit   OSLO_NEW_CARS benefit:0000        2     2026-04-30 UPSERT Re43073e4a0c5    item0_base   1017      USD
benefit   OSLO_NEW_CARS benefit:0021        1     2026-04-02 UPSERT R2ff8218f0845 item10_rebate  -4893      JPY
benefit   OSLO_NEW_CARS benefit:0036        2     2026-04-30 UPSERT R8efb61b6c33b   item18_base  18074      EUR
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 23}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 19}, "component": {"missing": 0, "unique_nonempty": 23}, "amount": {"missing": 0, "unique_nonempty": 41, "numeric_range": [-4900.0, 126507.0]}, "currency": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "benefit", "matching_columns": [{"column": "table", "exact": 49, "prefix": 49, "contains": 49, "examples": ["benefit"]}, {"column": "record_id", "exact": 0, "prefix": 49, "contains": 49, "examples": ["benefit:0039", "benefit:0018", "benefit:0009"]}], "exact_matching_columns": 1}, {"term": "currency", "matching_columns": [], "exact_matching_columns": 0}, {"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 45, "prefix": 45, "contains": 45, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usd", "matching_columns": [{"column": "currency", "exact": 30, "prefix": 30, "contains": 30, "examples": ["USD"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_04/export_04.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 13
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_a', 'item_b', 'bonus_cents']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_a': 'object', 'item_b': 'object', 'bonus_cents': 'float64'}
Preview only (first 10 rows):
 table dealership_id        record_id revision effective_date action        item_a        item_b bonus_cents
bundle OSLO_NEW_CARS      bundle:0000        1     2026-04-02 UPSERT R144a3b2dac92 R45289ace3322         448
bundle OSLO_NEW_CARS      bundle:0006        2     2026-04-30 UPSERT R6121e8a8833d R4e00b9c54be8         404
bundle OSLO_NEW_CARS      bundle:0002        1     2026-04-02 UPSERT R8efb61b6c33b R1295107ec030         441
bundle OSLO_NEW_CARS      bundle:0006        1     2026-04-02 UPSERT R6121e8a8833d R4e00b9c54be8         411
bundle OSLO_NEW_CARS      bundle:0004        1     2026-04-02 UPSERT Rc6cd5cbe3d08 Rf9d792d5328c         173
bundle OSLO_NEW_CARS bundle:withdrawn        1     2026-04-02 UPSERT R144a3b2dac92 R45289ace3322         441
bundle OSLO_NEW_CARS      bundle:0000        2     2026-04-30 UPSERT R144a3b2dac92 R45289ace3322         441
bundle OSLO_NEW_CARS      bundle:0000        2     2026-04-30 UPSERT R144a3b2dac92 R45289ace3322         441
bundle OSLO_NEW_CARS bundle:withdrawn        2     2026-05-05 DELETE                                        
bundle OSLO_NEW_CARS      bundle:0004        2     2026-04-30 UPSERT Rc6cd5cbe3d08 Rf9d792d5328c         166
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 5}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_a": {"missing": 1, "unique_nonempty": 4}, "item_b": {"missing": 1, "unique_nonempty": 4}, "bonus_cents": {"missing": 1, "unique_nonempty": 7, "numeric_range": [166.0, 448.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "bundle", "matching_columns": [{"column": "table", "exact": 13, "prefix": 13, "contains": 13, "examples": ["bundle"]}, {"column": "record_id", "exact": 0, "prefix": 13, "contains": 13, "examples": ["bundle:0000", "bundle:0006", "bundle:0002"]}], "exact_matching_columns": 1}, {"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 12, "prefix": 12, "contains": 12, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_05/export_05.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 11
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_a', 'item_b', 'bonus_cents']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_a': 'object', 'item_b': 'object', 'bonus_cents': 'int64'}
Preview only (first 10 rows):
 table   dealership_id   record_id revision effective_date action        item_a        item_b bonus_cents
bundle   OSLO_NEW_CARS bundle:0007        1     2026-04-02 UPSERT R3127c4fdc4b8 R29226c016297         268
bundle   OSLO_NEW_CARS bundle:0005        1     2026-04-02 UPSERT R45289ace3322 Rf9d792d5328c         108
bundle   OSLO_NEW_CARS bundle:0007        2     2026-04-30 UPSERT R3127c4fdc4b8 R29226c016297         261
bundle   OSLO_NEW_CARS bundle:0003        2     2026-04-30 UPSERT R17fb1a9043c3 R4e00b9c54be8         359
bundle   OSLO_NEW_CARS bundle:0003        1     2026-04-02 UPSERT R17fb1a9043c3 R4e00b9c54be8         366
bundle   OSLO_NEW_CARS bundle:0001        1     2026-04-02 UPSERT R29226c016297 Rf9d792d5328c         429
bundle   OSLO_NEW_CARS bundle:0005        3     2026-05-19 UPSERT R45289ace3322 Rf9d792d5328c         108
bundle BERGEN_NEW_CARS bundle:0004        2     2026-04-30 UPSERT Rc6cd5cbe3d08 Rf9d792d5328c         166
bundle   OSLO_NEW_CARS bundle:0001        2     2026-04-30 UPSERT R29226c016297 Rf9d792d5328c         422
bundle   OSLO_NEW_CARS bundle:0005        2     2026-04-30 UPSERT R45289ace3322 Rf9d792d5328c         101
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 5}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 5}, "item_b": {"missing": 0, "unique_nonempty": 3}, "bonus_cents": {"missing": 0, "unique_nonempty": 9, "numeric_range": [101.0, 429.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "bundle", "matching_columns": [{"column": "table", "exact": 11, "prefix": 11, "contains": 11, "examples": ["bundle"]}, {"column": "record_id", "exact": 0, "prefix": 11, "contains": 11, "examples": ["bundle:0007", "bundle:0005", "bundle:0003"]}], "exact_matching_columns": 1}, {"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 10, "prefix": 10, "contains": 10, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_06/export_06.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 11
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'resource', 'entry', 'amount', 'unit']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'resource': 'object', 'entry': 'object', 'amount': 'float64', 'unit': 'object'}
Preview only (first 10 rows):
          table   dealership_id                 record_id revision effective_date action resource   entry amount   unit
capacity_ledger   OSLO_NEW_CARS capacity_ledger:withdrawn        2     2026-05-05 DELETE                               
capacity_ledger   OSLO_NEW_CARS      capacity_ledger:0000        1     2026-04-02 UPSERT    space opening 244007     ml
capacity_ledger   OSLO_NEW_CARS      capacity_ledger:0004        1     2026-04-02 UPSERT    power opening 298007     wh
capacity_ledger   OSLO_NEW_CARS      capacity_ledger:0002        1     2026-04-02 UPSERT    labor opening  12727 minute
capacity_ledger   OSLO_NEW_CARS      capacity_ledger:0000        3     2026-05-19 UPSERT    space opening 244007     ml
capacity_ledger   OSLO_NEW_CARS      capacity_ledger:0000        2     2026-04-30 UPSERT    space opening 244000     ml
capacity_ledger BERGEN_NEW_CARS      capacity_ledger:0000        2     2026-04-30 UPSERT    space opening 244000     ml
capacity_ledger   OSLO_NEW_CARS      capacity_ledger:0002        2     2026-04-30 UPSERT    labor opening  12720 minute
capacity_ledger   OSLO_NEW_CARS capacity_ledger:withdrawn        1     2026-04-02 UPSERT    space opening 244000     ml
capacity_ledger   OSLO_NEW_CARS      capacity_ledger:0004        2     2026-04-30 UPSERT    power opening 298000     wh
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 4}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "resource": {"missing": 1, "unique_nonempty": 3}, "entry": {"missing": 1, "unique_nonempty": 1}, "amount": {"missing": 1, "unique_nonempty": 6, "numeric_range": [12720.0, 298007.0]}, "unit": {"missing": 1, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity_ledger", "matching_columns": [{"column": "table", "exact": 11, "prefix": 11, "contains": 11, "examples": ["capacity_ledger"]}, {"column": "record_id", "exact": 0, "prefix": 11, "contains": 11, "examples": ["capacity_ledger:withdrawn", "capacity_ledger:0000", "capacity_ledger:0004"]}], "exact_matching_columns": 1}, {"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "ml", "matching_columns": [{"column": "unit", "exact": 6, "prefix": 6, "contains": 6, "examples": ["ml"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 10, "prefix": 10, "contains": 10, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "wh", "matching_columns": [{"column": "unit", "exact": 2, "prefix": 2, "contains": 2, "examples": ["wh"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_01/export_07.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'resource', 'entry', 'amount', 'unit']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'resource': 'object', 'entry': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
          table   dealership_id            record_id revision effective_date action resource       entry amount   unit
capacity_ledger   OSLO_NEW_CARS capacity_ledger:0001        2     2026-04-30 UPSERT    space reservation  -9000     ml
capacity_ledger   OSLO_NEW_CARS capacity_ledger:0003        2     2026-04-30 UPSERT    labor reservation   -420 minute
capacity_ledger   OSLO_NEW_CARS capacity_ledger:0005        1     2026-04-02 UPSERT    power reservation -10993     wh
capacity_ledger   OSLO_NEW_CARS capacity_ledger:0003        1     2026-04-02 UPSERT    labor reservation   -413 minute
capacity_ledger   OSLO_NEW_CARS capacity_ledger:0001        1     2026-04-02 UPSERT    space reservation  -8993     ml
capacity_ledger BERGEN_NEW_CARS capacity_ledger:0004        2     2026-04-30 UPSERT    power     opening 298000     wh
capacity_ledger   OSLO_NEW_CARS capacity_ledger:0005        3     2026-05-19 UPSERT    power reservation -10993     wh
capacity_ledger   OSLO_NEW_CARS capacity_ledger:0005        2     2026-04-30 UPSERT    power reservation -11000     wh
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 4}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "resource": {"missing": 0, "unique_nonempty": 3}, "entry": {"missing": 0, "unique_nonempty": 2}, "amount": {"missing": 0, "unique_nonempty": 7, "numeric_range": [-11000.0, 298000.0]}, "unit": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity_ledger", "matching_columns": [{"column": "table", "exact": 8, "prefix": 8, "contains": 8, "examples": ["capacity_ledger"]}, {"column": "record_id", "exact": 0, "prefix": 8, "contains": 8, "examples": ["capacity_ledger:0001", "capacity_ledger:0003", "capacity_ledger:0005"]}], "exact_matching_columns": 1}, {"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "ml", "matching_columns": [{"column": "unit", "exact": 2, "prefix": 2, "contains": 2, "examples": ["ml"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 7, "prefix": 7, "contains": 7, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "wh", "matching_columns": [{"column": "unit", "exact": 4, "prefix": 4, "contains": 4, "examples": ["wh"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_02/export_08.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'category': 'object', 'minimum_quantity': 'int64', 'maximum_quantity': 'int64', 'activation_fee_cents': 'int64'}
Preview only (first 10 rows):
   table   dealership_id     record_id revision effective_date action category minimum_quantity maximum_quantity activation_fee_cents
category   OSLO_NEW_CARS category:0003        2     2026-04-30 UPSERT       G3                9               21                  349
category BERGEN_NEW_CARS category:0000        2     2026-04-30 UPSERT       G0                9               21                  333
category   OSLO_NEW_CARS category:0003        1     2026-04-02 UPSERT       G3               16               28                  356
category   OSLO_NEW_CARS category:0001        1     2026-04-02 UPSERT       G1               16               28                  394
category   OSLO_NEW_CARS category:0001        2     2026-04-30 UPSERT       G1                9               21                  387
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 3}, "revision": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.0, 2.0]}, "effective_date": {"missing": 0, "unique_nonempty": 2}, "action": {"missing": 0, "unique_nonempty": 1}, "category": {"missing": 0, "unique_nonempty": 3}, "minimum_quantity": {"missing": 0, "unique_nonempty": 2, "numeric_range": [9.0, 16.0]}, "maximum_quantity": {"missing": 0, "unique_nonempty": 2, "numeric_range": [21.0, 28.0]}, "activation_fee_cents": {"missing": 0, "unique_nonempty": 5, "numeric_range": [333.0, 394.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "category", "matching_columns": [{"column": "table", "exact": 5, "prefix": 5, "contains": 5, "examples": ["category"]}, {"column": "record_id", "exact": 0, "prefix": 5, "contains": 5, "examples": ["category:0003", "category:0000", "category:0001"]}], "exact_matching_columns": 1}, {"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 4, "prefix": 4, "contains": 4, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_03/export_09.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'category': 'object', 'minimum_quantity': 'float64', 'maximum_quantity': 'float64', 'activation_fee_cents': 'float64'}
Preview only (first 10 rows):
   table dealership_id          record_id revision effective_date action category minimum_quantity maximum_quantity activation_fee_cents
category OSLO_NEW_CARS category:withdrawn        2     2026-05-05 DELETE                                                                
category OSLO_NEW_CARS      category:0000        2     2026-04-30 UPSERT       G0                9               21                  333
category OSLO_NEW_CARS      category:0000        1     2026-04-02 UPSERT       G0               16               28                  340
category OSLO_NEW_CARS      category:0002        1     2026-04-02 UPSERT       G2               12               24                  230
category OSLO_NEW_CARS      category:0002        2     2026-04-30 UPSERT       G2                5               17                  223
category OSLO_NEW_CARS category:withdrawn        1     2026-04-02 UPSERT       G0                9               21                  333
category OSLO_NEW_CARS      category:0000        2     2026-04-30 UPSERT       G0                9               21                  333
category OSLO_NEW_CARS      category:0000        3     2026-05-19 UPSERT       G0               16               28                  340
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 1}, "record_id": {"missing": 0, "unique_nonempty": 3}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "category": {"missing": 1, "unique_nonempty": 2}, "minimum_quantity": {"missing": 1, "unique_nonempty": 4, "numeric_range": [5.0, 16.0]}, "maximum_quantity": {"missing": 1, "unique_nonempty": 4, "numeric_range": [17.0, 28.0]}, "activation_fee_cents": {"missing": 1, "unique_nonempty": 4, "numeric_range": [223.0, 340.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "category", "matching_columns": [{"column": "table", "exact": 8, "prefix": 8, "contains": 8, "examples": ["category"]}, {"column": "record_id", "exact": 0, "prefix": 8, "contains": 8, "examples": ["category:withdrawn", "category:0000", "category:0002"]}], "exact_matching_columns": 1}, {"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 8, "prefix": 8, "contains": 8, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_04/export_10.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 11
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'currency', 'usd_cents_numerator', 'denominator']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'currency': 'object', 'usd_cents_numerator': 'float64', 'denominator': 'float64'}
Preview only (first 10 rows):
table   dealership_id    record_id revision effective_date action currency usd_cents_numerator denominator
   fx   OSLO_NEW_CARS      fx:0001        2     2026-04-30 UPSERT      EUR                   1           2
   fx   OSLO_NEW_CARS      fx:0000        2     2026-04-30 UPSERT      USD                   1           1
   fx   OSLO_NEW_CARS      fx:0002        1     2026-04-02 UPSERT      JPY                   8         107
   fx   OSLO_NEW_CARS      fx:0000        3     2026-05-19 UPSERT      USD                   8           8
   fx   OSLO_NEW_CARS fx:withdrawn        1     2026-04-02 UPSERT      USD                   1           1
   fx   OSLO_NEW_CARS      fx:0001        1     2026-04-02 UPSERT      EUR                   8           9
   fx   OSLO_NEW_CARS      fx:0000        1     2026-04-02 UPSERT      USD                   8           8
   fx   OSLO_NEW_CARS      fx:0000        2     2026-04-30 UPSERT      USD                   1           1
   fx   OSLO_NEW_CARS fx:withdrawn        2     2026-05-05 DELETE                                         
   fx BERGEN_NEW_CARS      fx:0000        2     2026-04-30 UPSERT      USD                   1           1
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 4}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "currency": {"missing": 1, "unique_nonempty": 3}, "usd_cents_numerator": {"missing": 1, "unique_nonempty": 2, "numeric_range": [1.0, 8.0]}, "denominator": {"missing": 1, "unique_nonempty": 6, "numeric_range": [1.0, 107.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "currency", "matching_columns": [], "exact_matching_columns": 0}, {"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "denominator", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fx", "matching_columns": [{"column": "table", "exact": 11, "prefix": 11, "contains": 11, "examples": ["fx"]}, {"column": "record_id", "exact": 0, "prefix": 11, "contains": 11, "examples": ["fx:0001", "fx:0000", "fx:0002"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 10, "prefix": 10, "contains": 10, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usd", "matching_columns": [{"column": "currency", "exact": 6, "prefix": 6, "contains": 6, "examples": ["USD"]}], "exact_matching_columns": 1}, {"term": "usd_cents_numerator", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_05/export_11.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 37
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'ref', 'kind', 'entity_id', 'display_name']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'ref': 'object', 'kind': 'object', 'entity_id': 'object', 'display_name': 'object'}
Preview only (first 10 rows):
   table   dealership_id     record_id revision effective_date action           ref kind     entity_id    display_name
identity   OSLO_NEW_CARS identity:0003        1     2026-04-02 UPSERT Rdd7cda4c3010 item RA4_OPTION_04 Hybrid Vehicles
identity   OSLO_NEW_CARS identity:0017        2     2026-04-30 UPSERT Re8217731b8ed item RA4_OPTION_18   Luxury Sedans
identity   OSLO_NEW_CARS identity:0015        3     2026-05-19 UPSERT Rbbac12e00a0c item RA4_OPTION_16     Sports Cars
identity   OSLO_NEW_CARS identity:0001        1     2026-04-02 UPSERT Rb981c8b74356 item RA4_OPTION_02            SUVs
identity   OSLO_NEW_CARS identity:0023        1     2026-04-02 UPSERT R5177e1b63b51 item RA4_OPTION_24 Hybrid Vehicles
identity   OSLO_NEW_CARS identity:0027        1     2026-04-02 UPSERT R3127c4fdc4b8 item RA4_OPTION_28   Luxury Sedans
identity   OSLO_NEW_CARS identity:0009        2     2026-04-30 UPSERT R826527278104 item RA4_OPTION_10   Pickup Trucks
identity   OSLO_NEW_CARS identity:0005        1     2026-04-02 UPSERT Rb30197d6899f item RA4_OPTION_06     Sports Cars
identity BERGEN_NEW_CARS identity:0016        2     2026-04-30 UPSERT R17fb1a9043c3 item RA4_OPTION_17    Compact Cars
identity   OSLO_NEW_CARS identity:0021        2     2026-04-30 UPSERT R3c2a89004af3 item RA4_OPTION_22            SUVs
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 18}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "ref": {"missing": 0, "unique_nonempty": 18}, "kind": {"missing": 0, "unique_nonempty": 1}, "entity_id": {"missing": 0, "unique_nonempty": 18}, "display_name": {"missing": 0, "unique_nonempty": 9}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item", "matching_columns": [{"column": "kind", "exact": 37, "prefix": 37, "contains": 37, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 33, "prefix": 33, "contains": 33, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_06/export_12.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 38
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'ref', 'kind', 'entity_id', 'display_name']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'ref': 'object', 'kind': 'object', 'entity_id': 'object', 'display_name': 'object'}
Preview only (first 10 rows):
   table   dealership_id     record_id revision effective_date action           ref kind     entity_id      display_name
identity BERGEN_NEW_CARS identity:0020        2     2026-04-30 UPSERT R1295107ec030 item RA4_OPTION_21            Sedans
identity   OSLO_NEW_CARS identity:0002        1     2026-04-02 UPSERT R2dd6dc93369f item RA4_OPTION_03 Electric Vehicles
identity   OSLO_NEW_CARS identity:0022        2     2026-04-30 UPSERT R30a2c98896e0 item RA4_OPTION_23 Electric Vehicles
identity   OSLO_NEW_CARS identity:0016        2     2026-04-30 UPSERT R17fb1a9043c3 item RA4_OPTION_17      Compact Cars
identity   OSLO_NEW_CARS identity:0008        2     2026-04-30 UPSERT R4e00b9c54be8 item RA4_OPTION_09              Vans
identity   OSLO_NEW_CARS identity:0004        2     2026-04-30 UPSERT R6121e8a8833d item RA4_OPTION_05            Trucks
identity   OSLO_NEW_CARS identity:0024        2     2026-04-30 UPSERT R45289ace3322 item RA4_OPTION_25            Trucks
identity BERGEN_NEW_CARS identity:0012        2     2026-04-30 UPSERT R144a3b2dac92 item RA4_OPTION_13 Electric Vehicles
identity BERGEN_NEW_CARS identity:0004        2     2026-04-30 UPSERT R6121e8a8833d item RA4_OPTION_05            Trucks
identity   OSLO_NEW_CARS identity:0024        1     2026-04-02 UPSERT R45289ace3322 item RA4_OPTION_25            Trucks
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 15}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "ref": {"missing": 1, "unique_nonempty": 14}, "kind": {"missing": 1, "unique_nonempty": 1}, "entity_id": {"missing": 1, "unique_nonempty": 14}, "display_name": {"missing": 1, "unique_nonempty": 5}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "item", "matching_columns": [{"column": "kind", "exact": 37, "prefix": 37, "contains": 37, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 35, "prefix": 35, "contains": 35, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_01/export_13.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 16
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_a', 'item_b']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_a': 'object', 'item_b': 'object'}
Preview only (first 10 rows):
       table   dealership_id         record_id revision effective_date action        item_a        item_b
incompatible   OSLO_NEW_CARS incompatible:0007        2     2026-04-30 UPSERT R5177e1b63b51 R29226c016297
incompatible   OSLO_NEW_CARS incompatible:0005        2     2026-04-30 UPSERT R2ff8218f0845 Re8217731b8ed
incompatible   OSLO_NEW_CARS incompatible:0007        1     2026-04-02 UPSERT R5177e1b63b51 R29226c016297
incompatible BERGEN_NEW_CARS incompatible:0000        2     2026-04-30 UPSERT R2dd6dc93369f R144a3b2dac92
incompatible   OSLO_NEW_CARS incompatible:0007        2     2026-04-30 UPSERT R5177e1b63b51 R29226c016297
incompatible   OSLO_NEW_CARS incompatible:0009        1     2026-04-02 UPSERT Re4cddba9a9d6 R26a5710f69ac
incompatible   OSLO_NEW_CARS incompatible:0001        2     2026-04-30 UPSERT R777716c59761 R29226c016297
incompatible   OSLO_NEW_CARS incompatible:0009        2     2026-04-30 UPSERT Re4cddba9a9d6 R26a5710f69ac
incompatible   OSLO_NEW_CARS incompatible:0011        2     2026-04-30 UPSERT R1295107ec030 Rc6cd5cbe3d08
incompatible   OSLO_NEW_CARS incompatible:0005        3     2026-05-19 UPSERT R2ff8218f0845 Re8217731b8ed
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 8}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 7}, "item_b": {"missing": 0, "unique_nonempty": 7}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "incompatible", "matching_columns": [{"column": "table", "exact": 16, "prefix": 16, "contains": 16, "examples": ["incompatible"]}, {"column": "record_id", "exact": 0, "prefix": 16, "contains": 16, "examples": ["incompatible:0007", "incompatible:0005", "incompatible:0000"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 14, "prefix": 14, "contains": 14, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_02/export_14.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 18
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_a', 'item_b']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_a': 'object', 'item_b': 'object'}
Preview only (first 10 rows):
       table dealership_id              record_id revision effective_date action        item_a        item_b
incompatible OSLO_NEW_CARS      incompatible:0004        1     2026-04-02 UPSERT Rbbac12e00a0c Rc6cd5cbe3d08
incompatible OSLO_NEW_CARS incompatible:withdrawn        2     2026-05-05 DELETE                            
incompatible OSLO_NEW_CARS      incompatible:0006        1     2026-04-02 UPSERT Re4cddba9a9d6 R2dd6dc93369f
incompatible OSLO_NEW_CARS      incompatible:0002        1     2026-04-02 UPSERT R2ff8218f0845 R1295107ec030
incompatible OSLO_NEW_CARS      incompatible:0006        2     2026-04-30 UPSERT Re4cddba9a9d6 R2dd6dc93369f
incompatible OSLO_NEW_CARS      incompatible:0002        2     2026-04-30 UPSERT R2ff8218f0845 R1295107ec030
incompatible OSLO_NEW_CARS      incompatible:0000        3     2026-05-19 UPSERT R2dd6dc93369f R144a3b2dac92
incompatible OSLO_NEW_CARS incompatible:withdrawn        1     2026-04-02 UPSERT R2dd6dc93369f R144a3b2dac92
incompatible OSLO_NEW_CARS      incompatible:0008        1     2026-04-02 UPSERT R45289ace3322 R826527278104
incompatible OSLO_NEW_CARS      incompatible:0004        2     2026-04-30 UPSERT Rbbac12e00a0c Rc6cd5cbe3d08
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 7}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_a": {"missing": 1, "unique_nonempty": 6}, "item_b": {"missing": 1, "unique_nonempty": 6}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "incompatible", "matching_columns": [{"column": "table", "exact": 18, "prefix": 18, "contains": 18, "examples": ["incompatible"]}, {"column": "record_id", "exact": 0, "prefix": 18, "contains": 18, "examples": ["incompatible:0004", "incompatible:withdrawn", "incompatible:0006"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 17, "prefix": 17, "contains": 17, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_03/export_15.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 37
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'configuration_id']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'category': 'object', 'authorized': 'int64', 'minimum_lot': 'int64', 'maximum_order': 'int64', 'configuration_id': 'object'}
Preview only (first 10 rows):
table   dealership_id record_id revision effective_date action      item_ref category authorized minimum_lot maximum_order configuration_id
 item   OSLO_NEW_CARS item:0005        3     2026-05-19 UPSERT Rb30197d6899f       G1          8           9            19           CFG_01
 item   OSLO_NEW_CARS item:0023        1     2026-04-02 UPSERT R5177e1b63b51       G3          8           9            20           CFG_03
 item   OSLO_NEW_CARS item:0025        2     2026-04-30 UPSERT R29226c016297       G1          1           2            12           CFG_03
 item BERGEN_NEW_CARS item:0000        2     2026-04-30 UPSERT Re43073e4a0c5       G0          1           2            11           CFG_01
 item   OSLO_NEW_CARS item:0003        1     2026-04-02 UPSERT Rdd7cda4c3010       G3          8           9            20           CFG_01
 item   OSLO_NEW_CARS item:0007        2     2026-04-30 UPSERT Rc6cd5cbe3d08       G3          1           2            10           CFG_01
 item   OSLO_NEW_CARS item:0001        2     2026-04-30 UPSERT Rb981c8b74356       G1          1           2            12           CFG_01
 item   OSLO_NEW_CARS item:0005        2     2026-04-30 UPSERT Rb30197d6899f       G1          1           2            12           CFG_01
 item   OSLO_NEW_CARS item:0011        1     2026-04-02 UPSERT Rf9d792d5328c       G3          8           9            19           CFG_02
 item   OSLO_NEW_CARS item:0025        3     2026-05-19 UPSERT R29226c016297       G1          8           9            19           CFG_03
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 18}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 18}, "category": {"missing": 0, "unique_nonempty": 3}, "authorized": {"missing": 0, "unique_nonempty": 4, "numeric_range": [0.0, 8.0]}, "minimum_lot": {"missing": 0, "unique_nonempty": 2, "numeric_range": [2.0, 9.0]}, "maximum_order": {"missing": 0, "unique_nonempty": 12, "numeric_range": [7.0, 20.0]}, "configuration_id": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "authorized", "matching_columns": [], "exact_matching_columns": 0}, {"term": "category", "matching_columns": [], "exact_matching_columns": 0}, {"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item", "matching_columns": [{"column": "table", "exact": 37, "prefix": 37, "contains": 37, "examples": ["item"]}, {"column": "record_id", "exact": 0, "prefix": 37, "contains": 37, "examples": ["item:0005", "item:0023", "item:0025"]}], "exact_matching_columns": 1}, {"term": "maximum_order", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 33, "prefix": 33, "contains": 33, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_04/export_16.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 38
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'configuration_id']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'category': 'object', 'authorized': 'float64', 'minimum_lot': 'float64', 'maximum_order': 'float64', 'configuration_id': 'object'}
Preview only (first 10 rows):
table dealership_id record_id revision effective_date action      item_ref category authorized minimum_lot maximum_order configuration_id
 item OSLO_NEW_CARS item:0002        1     2026-04-02 UPSERT R2dd6dc93369f       G2          8           9            14           CFG_01
 item OSLO_NEW_CARS item:0018        1     2026-04-02 UPSERT R8efb61b6c33b       G2          7           9            19           CFG_02
 item OSLO_NEW_CARS item:0016        1     2026-04-02 UPSERT R17fb1a9043c3       G0          8           9            15           CFG_02
 item OSLO_NEW_CARS item:0018        2     2026-04-30 UPSERT R8efb61b6c33b       G2          0           2            12           CFG_02
 item OSLO_NEW_CARS item:0012        1     2026-04-02 UPSERT R144a3b2dac92       G0          8           9            17           CFG_02
 item OSLO_NEW_CARS item:0004        1     2026-04-02 UPSERT R6121e8a8833d       G0          8           9            14           CFG_01
 item OSLO_NEW_CARS item:0020        2     2026-04-30 UPSERT R1295107ec030       G0          1           2            11           CFG_03
 item OSLO_NEW_CARS item:0020        1     2026-04-02 UPSERT R1295107ec030       G0          8           9            18           CFG_03
 item OSLO_NEW_CARS item:0024        2     2026-04-30 UPSERT R45289ace3322       G0          0           2            12           CFG_03
 item OSLO_NEW_CARS item:0022        1     2026-04-02 UPSERT R30a2c98896e0       G2          8           9            17           CFG_03
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 15}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_ref": {"missing": 1, "unique_nonempty": 14}, "category": {"missing": 1, "unique_nonempty": 2}, "authorized": {"missing": 1, "unique_nonempty": 4, "numeric_range": [0.0, 8.0]}, "minimum_lot": {"missing": 1, "unique_nonempty": 2, "numeric_range": [2.0, 9.0]}, "maximum_order": {"missing": 1, "unique_nonempty": 10, "numeric_range": [7.0, 19.0]}, "configuration_id": {"missing": 1, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "authorized", "matching_columns": [], "exact_matching_columns": 0}, {"term": "category", "matching_columns": [], "exact_matching_columns": 0}, {"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "item", "matching_columns": [{"column": "table", "exact": 38, "prefix": 38, "contains": 38, "examples": ["item"]}, {"column": "record_id", "exact": 0, "prefix": 38, "contains": 38, "examples": ["item:0002", "item:0018", "item:0016"]}], "exact_matching_columns": 1}, {"term": "maximum_order", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 35, "prefix": 35, "contains": 35, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_05/export_17.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 37
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'activation_fee_cents': 'int64'}
Preview only (first 10 rows):
   table   dealership_id     record_id revision effective_date action      item_ref activation_fee_cents
item_fee BERGEN_NEW_CARS item_fee:0016        2     2026-04-30 UPSERT R17fb1a9043c3                  260
item_fee   OSLO_NEW_CARS item_fee:0013        2     2026-04-30 UPSERT Re4cddba9a9d6                  273
item_fee   OSLO_NEW_CARS item_fee:0007        2     2026-04-30 UPSERT Rc6cd5cbe3d08                  257
item_fee   OSLO_NEW_CARS item_fee:0021        2     2026-04-30 UPSERT R3c2a89004af3                  151
item_fee   OSLO_NEW_CARS item_fee:0025        1     2026-04-02 UPSERT R29226c016297                  304
item_fee   OSLO_NEW_CARS item_fee:0021        1     2026-04-02 UPSERT R3c2a89004af3                  158
item_fee   OSLO_NEW_CARS item_fee:0017        2     2026-04-30 UPSERT Re8217731b8ed                  442
item_fee   OSLO_NEW_CARS item_fee:0027        1     2026-04-02 UPSERT R3127c4fdc4b8                  220
item_fee   OSLO_NEW_CARS item_fee:0027        2     2026-04-30 UPSERT R3127c4fdc4b8                  213
item_fee   OSLO_NEW_CARS item_fee:0009        2     2026-04-30 UPSERT R826527278104                  186
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 18}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 18}, "activation_fee_cents": {"missing": 0, "unique_nonempty": 30, "numeric_range": [115.0, 449.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item_fee", "matching_columns": [{"column": "table", "exact": 37, "prefix": 37, "contains": 37, "examples": ["item_fee"]}, {"column": "record_id", "exact": 0, "prefix": 37, "contains": 37, "examples": ["item_fee:0016", "item_fee:0013", "item_fee:0007"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 33, "prefix": 33, "contains": 33, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_06/export_18.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 38
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'activation_fee_cents': 'float64'}
Preview only (first 10 rows):
   table   dealership_id     record_id revision effective_date action      item_ref activation_fee_cents
item_fee   OSLO_NEW_CARS item_fee:0016        2     2026-04-30 UPSERT R17fb1a9043c3                  260
item_fee   OSLO_NEW_CARS item_fee:0008        2     2026-04-30 UPSERT R4e00b9c54be8                  256
item_fee   OSLO_NEW_CARS item_fee:0016        1     2026-04-02 UPSERT R17fb1a9043c3                  267
item_fee   OSLO_NEW_CARS item_fee:0002        1     2026-04-02 UPSERT R2dd6dc93369f                  135
item_fee   OSLO_NEW_CARS item_fee:0020        1     2026-04-02 UPSERT R1295107ec030                  294
item_fee   OSLO_NEW_CARS item_fee:0022        2     2026-04-30 UPSERT R30a2c98896e0                  366
item_fee   OSLO_NEW_CARS item_fee:0002        2     2026-04-30 UPSERT R2dd6dc93369f                  128
item_fee   OSLO_NEW_CARS item_fee:0000        2     2026-04-30 UPSERT Re43073e4a0c5                  210
item_fee BERGEN_NEW_CARS item_fee:0004        2     2026-04-30 UPSERT R6121e8a8833d                  363
item_fee   OSLO_NEW_CARS item_fee:0018        1     2026-04-02 UPSERT R8efb61b6c33b                  367
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 15}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_ref": {"missing": 1, "unique_nonempty": 14}, "activation_fee_cents": {"missing": 1, "unique_nonempty": 28, "numeric_range": [128.0, 498.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "item_fee", "matching_columns": [{"column": "table", "exact": 38, "prefix": 38, "contains": 38, "examples": ["item_fee"]}, {"column": "record_id", "exact": 0, "prefix": 38, "contains": 38, "examples": ["item_fee:0016", "item_fee:0008", "item_fee:0002"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 35, "prefix": 35, "contains": 35, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_01/export_19.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 37
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'historical_sales', 'forecast_units', 'margin_percent']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'historical_sales': 'int64', 'forecast_units': 'int64', 'margin_percent': 'int64'}
Preview only (first 10 rows):
 table   dealership_id   record_id revision effective_date action      item_ref historical_sales forecast_units margin_percent
market   OSLO_NEW_CARS market:0015        2     2026-04-30 UPSERT Rbbac12e00a0c                2              1             16
market   OSLO_NEW_CARS market:0019        1     2026-04-02 UPSERT R777716c59761                9             10             30
market   OSLO_NEW_CARS market:0007        2     2026-04-30 UPSERT Rc6cd5cbe3d08                3              1             17
market   OSLO_NEW_CARS market:0027        1     2026-04-02 UPSERT R3127c4fdc4b8               10             10             41
market   OSLO_NEW_CARS market:0003        2     2026-04-30 UPSERT Rdd7cda4c3010                1              2             24
market   OSLO_NEW_CARS market:0019        2     2026-04-30 UPSERT R777716c59761                2              3             23
market   OSLO_NEW_CARS market:0005        1     2026-04-02 UPSERT Rb30197d6899f               10              8             26
market   OSLO_NEW_CARS market:0021        2     2026-04-30 UPSERT R3c2a89004af3                3              2             18
market   OSLO_NEW_CARS market:0025        1     2026-04-02 UPSERT R29226c016297               10             10             32
market BERGEN_NEW_CARS market:0024        2     2026-04-30 UPSERT R45289ace3322                3              2             11
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 18}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 18}, "historical_sales": {"missing": 0, "unique_nonempty": 6, "numeric_range": [1.0, 10.0]}, "forecast_units": {"missing": 0, "unique_nonempty": 6, "numeric_range": [1.0, 10.0]}, "margin_percent": {"missing": 0, "unique_nonempty": 22, "numeric_range": [5.0, 41.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 33, "prefix": 33, "contains": 33, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_02/export_20.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 38
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'historical_sales', 'forecast_units', 'margin_percent']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'historical_sales': 'float64', 'forecast_units': 'float64', 'margin_percent': 'float64'}
Preview only (first 10 rows):
 table dealership_id   record_id revision effective_date action      item_ref historical_sales forecast_units margin_percent
market OSLO_NEW_CARS market:0004        1     2026-04-02 UPSERT R6121e8a8833d               10              8             21
market OSLO_NEW_CARS market:0006        2     2026-04-30 UPSERT R26a5710f69ac                1              2             35
market OSLO_NEW_CARS market:0018        1     2026-04-02 UPSERT R8efb61b6c33b               10             10             28
market OSLO_NEW_CARS market:0004        2     2026-04-30 UPSERT R6121e8a8833d                3              1             14
market OSLO_NEW_CARS market:0026        1     2026-04-02 UPSERT R4523ce0a5d8d                9              8             23
market OSLO_NEW_CARS market:0010        2     2026-04-30 UPSERT R2ff8218f0845                3              2             30
market OSLO_NEW_CARS market:0022        1     2026-04-02 UPSERT R30a2c98896e0               10             10             40
market OSLO_NEW_CARS market:0016        1     2026-04-02 UPSERT R17fb1a9043c3                9             10             30
market OSLO_NEW_CARS market:0006        1     2026-04-02 UPSERT R26a5710f69ac                8              9             42
market OSLO_NEW_CARS market:0000        1     2026-04-02 UPSERT Re43073e4a0c5                9             10             29
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 15}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_ref": {"missing": 1, "unique_nonempty": 14}, "historical_sales": {"missing": 1, "unique_nonempty": 6, "numeric_range": [1.0, 10.0]}, "forecast_units": {"missing": 1, "unique_nonempty": 6, "numeric_range": [1.0, 10.0]}, "margin_percent": {"missing": 1, "unique_nonempty": 20, "numeric_range": [5.0, 42.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 35, "prefix": 35, "contains": 35, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_03/export_21.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 24
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'prerequisite_ref']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'prerequisite_ref': 'object'}
Preview only (first 10 rows):
   table   dealership_id          record_id revision effective_date action      item_ref prerequisite_ref
requires   OSLO_NEW_CARS      requires:0000        2     2026-04-30 UPSERT R144a3b2dac92    R26a5710f69ac
requires   OSLO_NEW_CARS      requires:0010        3     2026-05-19 UPSERT R30a2c98896e0    Rf9d792d5328c
requires   OSLO_NEW_CARS requires:withdrawn        2     2026-05-05 DELETE                               
requires   OSLO_NEW_CARS      requires:0012        1     2026-04-02 UPSERT R45289ace3322    R826527278104
requires BERGEN_NEW_CARS      requires:0008        2     2026-04-30 UPSERT R1295107ec030    Rc6cd5cbe3d08
requires   OSLO_NEW_CARS      requires:0002        2     2026-04-30 UPSERT Rde7df7983569    R2ff8218f0845
requires BERGEN_NEW_CARS      requires:0000        2     2026-04-30 UPSERT R144a3b2dac92    R26a5710f69ac
requires   OSLO_NEW_CARS      requires:0010        2     2026-04-30 UPSERT R30a2c98896e0    Rf9d792d5328c
requires   OSLO_NEW_CARS      requires:0000        1     2026-04-02 UPSERT R144a3b2dac92    R26a5710f69ac
requires   OSLO_NEW_CARS      requires:0000        3     2026-05-19 UPSERT R144a3b2dac92    R26a5710f69ac
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 9}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_ref": {"missing": 1, "unique_nonempty": 8}, "prerequisite_ref": {"missing": 1, "unique_nonempty": 7}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 22, "prefix": 22, "contains": 22, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "requires", "matching_columns": [{"column": "table", "exact": 24, "prefix": 24, "contains": 24, "examples": ["requires"]}, {"column": "record_id", "exact": 0, "prefix": 24, "contains": 24, "examples": ["requires:0000", "requires:0010", "requires:withdrawn"]}], "exact_matching_columns": 1}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_04/export_22.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 21
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'prerequisite_ref']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'prerequisite_ref': 'object'}
Preview only (first 10 rows):
   table   dealership_id     record_id revision effective_date action      item_ref prerequisite_ref
requires   OSLO_NEW_CARS requires:0005        2     2026-04-30 UPSERT Re8217731b8ed    Rdd7cda4c3010
requires   OSLO_NEW_CARS requires:0003        2     2026-04-30 UPSERT Rbbac12e00a0c    R826527278104
requires   OSLO_NEW_CARS requires:0007        2     2026-04-30 UPSERT R777716c59761    Rb981c8b74356
requires   OSLO_NEW_CARS requires:0001        1     2026-04-02 UPSERT Re4cddba9a9d6    Rc6cd5cbe3d08
requires   OSLO_NEW_CARS requires:0015        1     2026-04-02 UPSERT R3127c4fdc4b8    Rdd7cda4c3010
requires BERGEN_NEW_CARS requires:0004        2     2026-04-30 UPSERT R17fb1a9043c3    R2dd6dc93369f
requires   OSLO_NEW_CARS requires:0011        1     2026-04-02 UPSERT R5177e1b63b51    Re43073e4a0c5
requires   OSLO_NEW_CARS requires:0011        2     2026-04-30 UPSERT R5177e1b63b51    Re43073e4a0c5
requires   OSLO_NEW_CARS requires:0007        1     2026-04-02 UPSERT R777716c59761    Rb981c8b74356
requires   OSLO_NEW_CARS requires:0015        2     2026-04-30 UPSERT R3127c4fdc4b8    Rdd7cda4c3010
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 10}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 10}, "prerequisite_ref": {"missing": 0, "unique_nonempty": 6}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 19, "prefix": 19, "contains": 19, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "requires", "matching_columns": [{"column": "table", "exact": 21, "prefix": 21, "contains": 21, "examples": ["requires"]}, {"column": "record_id", "exact": 0, "prefix": 21, "contains": 21, "examples": ["requires:0005", "requires:0003", "requires:0007"]}], "exact_matching_columns": 1}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_05/export_23.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 75
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'float64', 'unit': 'object'}
Preview only (first 10 rows):
table dealership_id  record_id revision effective_date action      item_ref resource amount  unit
usage OSLO_NEW_CARS usage:0066        1     2026-04-02 UPSERT R30a2c98896e0    space     10 liter
usage OSLO_NEW_CARS usage:0045        3     2026-05-19 UPSERT Rbbac12e00a0c    space     11 liter
usage OSLO_NEW_CARS usage:0024        2     2026-04-30 UPSERT R4e00b9c54be8    space      2 liter
usage OSLO_NEW_CARS usage:0057        1     2026-04-02 UPSERT R777716c59761    space     12 liter
usage OSLO_NEW_CARS usage:0018        2     2026-04-30 UPSERT R26a5710f69ac    space      2 liter
usage OSLO_NEW_CARS usage:0012        2     2026-04-30 UPSERT R6121e8a8833d    space      5 liter
usage OSLO_NEW_CARS usage:0039        2     2026-04-30 UPSERT Re4cddba9a9d6    space      7 liter
usage OSLO_NEW_CARS usage:0000        3     2026-05-19 UPSERT Re43073e4a0c5    space     16 liter
usage OSLO_NEW_CARS usage:0063        1     2026-04-02 UPSERT R3c2a89004af3    space     15 liter
usage OSLO_NEW_CARS usage:0027        1     2026-04-02 UPSERT R826527278104    space     14 liter
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 29}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_ref": {"missing": 1, "unique_nonempty": 28}, "resource": {"missing": 1, "unique_nonempty": 1}, "amount": {"missing": 1, "unique_nonempty": 15, "numeric_range": [2.0, 16.0]}, "unit": {"missing": 1, "unique_nonempty": 1}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "liter", "matching_columns": [{"column": "unit", "exact": 74, "prefix": 74, "contains": 74, "examples": ["liter"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 68, "prefix": 68, "contains": 68, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 75, "prefix": 75, "contains": 75, "examples": ["usage"]}, {"column": "record_id", "exact": 0, "prefix": 75, "contains": 75, "examples": ["usage:0066", "usage:0045", "usage:0024"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_06/export_24.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 72
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
table dealership_id  record_id revision effective_date action      item_ref resource amount unit
usage OSLO_NEW_CARS usage:0004        2     2026-04-30 UPSERT Rb981c8b74356    labor      5 hour
usage OSLO_NEW_CARS usage:0028        2     2026-04-30 UPSERT R826527278104    labor      2 hour
usage OSLO_NEW_CARS usage:0082        1     2026-04-02 UPSERT R3127c4fdc4b8    labor     15 hour
usage OSLO_NEW_CARS usage:0070        2     2026-04-30 UPSERT R5177e1b63b51    labor      5 hour
usage OSLO_NEW_CARS usage:0043        2     2026-04-30 UPSERT Rde7df7983569    labor      6 hour
usage OSLO_NEW_CARS usage:0025        3     2026-05-19 UPSERT R4e00b9c54be8    labor     10 hour
usage OSLO_NEW_CARS usage:0016        1     2026-04-02 UPSERT Rb30197d6899f    labor     15 hour
usage OSLO_NEW_CARS usage:0067        2     2026-04-30 UPSERT R30a2c98896e0    labor      7 hour
usage OSLO_NEW_CARS usage:0073        1     2026-04-02 UPSERT R45289ace3322    labor     15 hour
usage OSLO_NEW_CARS usage:0040        2     2026-04-30 UPSERT Re4cddba9a9d6    labor      2 hour
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 28}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 28}, "resource": {"missing": 0, "unique_nonempty": 1}, "amount": {"missing": 0, "unique_nonempty": 13, "numeric_range": [2.0, 16.0]}, "unit": {"missing": 0, "unique_nonempty": 1}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "hour", "matching_columns": [{"column": "unit", "exact": 72, "prefix": 72, "contains": 72, "examples": ["hour"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 65, "prefix": 65, "contains": 65, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 72, "prefix": 72, "contains": 72, "examples": ["usage"]}, {"column": "record_id", "exact": 0, "prefix": 72, "contains": 72, "examples": ["usage:0004", "usage:0028", "usage:0082"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_01/export_25.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 73
Columns: ['table', 'dealership_id', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'dealership_id': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
table dealership_id  record_id revision effective_date action      item_ref resource amount unit
usage OSLO_NEW_CARS usage:0062        2     2026-04-30 UPSERT R1295107ec030    power      9  kwh
usage OSLO_NEW_CARS usage:0059        1     2026-04-02 UPSERT R777716c59761    power     16  kwh
usage OSLO_NEW_CARS usage:0056        2     2026-04-30 UPSERT R8efb61b6c33b    power      5  kwh
usage OSLO_NEW_CARS usage:0029        1     2026-04-02 UPSERT R826527278104    power     11  kwh
usage OSLO_NEW_CARS usage:0047        1     2026-04-02 UPSERT Rbbac12e00a0c    power     10  kwh
usage OSLO_NEW_CARS usage:0035        2     2026-04-30 UPSERT Rf9d792d5328c    power      9  kwh
usage OSLO_NEW_CARS usage:0065        1     2026-04-02 UPSERT R3c2a89004af3    power     11  kwh
usage OSLO_NEW_CARS usage:0011        1     2026-04-02 UPSERT Rdd7cda4c3010    power     13  kwh
usage OSLO_NEW_CARS usage:0041        2     2026-04-30 UPSERT Re4cddba9a9d6    power      6  kwh
usage OSLO_NEW_CARS usage:0047        2     2026-04-30 UPSERT Rbbac12e00a0c    power      3  kwh
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "dealership_id": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 28}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 28}, "resource": {"missing": 0, "unique_nonempty": 1}, "amount": {"missing": 0, "unique_nonempty": 15, "numeric_range": [2.0, 16.0]}, "unit": {"missing": 0, "unique_nonempty": 1}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "dealership_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "kwh", "matching_columns": [{"column": "unit", "exact": 73, "prefix": 73, "contains": 73, "examples": ["kwh"]}], "exact_matching_columns": 1}, {"term": "oslo_new_cars", "matching_columns": [{"column": "dealership_id", "exact": 66, "prefix": 66, "contains": 66, "examples": ["OSLO_NEW_CARS"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 73, "prefix": 73, "contains": 73, "examples": ["usage"]}, {"column": "record_id", "exact": 0, "prefix": 73, "contains": 73, "examples": ["usage:0062", "usage:0059", "usage:0056"]}], "exact_matching_columns": 1}]