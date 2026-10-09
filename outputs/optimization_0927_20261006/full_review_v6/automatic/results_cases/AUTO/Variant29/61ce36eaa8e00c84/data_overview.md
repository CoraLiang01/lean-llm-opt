File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_01.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 16
Columns: ['table', 'item_ref', 'prerequisite_ref']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'prerequisite_ref': 'object'}
Preview only (first 10 rows):
   table      item_ref prerequisite_ref
requires Re3f595d2bf06    R78298f0f56a2
requires R5dfabf4dd5b4    R189b5f29bad1
requires R0ae04f41cbb0    R436178c5636f
requires Rc4fc45bcfcbb    R189b5f29bad1
requires R56cb3a436a5a    R35897c8759f2
requires Rc3e636c6aeef    Rdce4df21cd05
requires R802e24035203    R35897c8759f2
requires Rd9d3a145aada    R189b5f29bad1
requires R19b312900bfe    R35897c8759f2
requires R640d4c97690d    R35897c8759f2
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 16}, "prerequisite_ref": {"missing": 0, "unique_nonempty": 6}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "requires", "matching_columns": [{"column": "table", "exact": 16, "prefix": 16, "contains": 16, "examples": ["requires"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_02.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 28
Columns: ['table', 'item_ref', 'historical_sales', 'forecast_units', 'margin_percent']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'historical_sales': 'int64', 'forecast_units': 'int64', 'margin_percent': 'int64'}
Preview only (first 10 rows):
 table      item_ref historical_sales forecast_units margin_percent
market R7e7dbfff0dc6                1              3             17
market Rc4fc45bcfcbb                2              1             23
market Rdce4df21cd05                1              1             13
market R35897c8759f2                1              3             22
market Rafc569d2d978                2              3             22
market R83c7f04f7b54                3              3             14
market R5dfabf4dd5b4                2              3             26
market Re3f595d2bf06                1              1             10
market R2efa1a0852bf                2              1             10
market R78298f0f56a2                1              2             21
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 28}, "historical_sales": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "forecast_units": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "margin_percent": {"missing": 0, "unique_nonempty": 17, "numeric_range": [6.0, 33.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: []

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_03.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 28
Columns: ['table', 'item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'unit_benefit_cents', 'item_fee_cents']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'category': 'object', 'authorized': 'int64', 'minimum_lot': 'int64', 'maximum_order': 'int64', 'unit_benefit_cents': 'int64', 'item_fee_cents': 'int64'}
Preview only (first 10 rows):
table      item_ref category authorized minimum_lot maximum_order unit_benefit_cents item_fee_cents
 item Re0a6ccc90076       G3          1           2            11                965            493
 item R35897c8759f2       G3          1           2            13               1569            200
 item R7e2a2c482e77       G3          0           2            12               9000            301
 item R7e7dbfff0dc6       G3          1           2            10               1074            121
 item Rd207cec92d9d       G3          1           2             7                349            258
 item R56cb3a436a5a       G1          0           2            12               9000            288
 item R9ec3bcfe9357       G2          1           2            12                278            274
 item R640d4c97690d       G2          1           2            13                281            157
 item Rd9d3a145aada       G3          1           2             8               1229            227
 item R19b312900bfe       G1          1           2            10               1578            292
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 28}, "category": {"missing": 0, "unique_nonempty": 4}, "authorized": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}, "minimum_lot": {"missing": 0, "unique_nonempty": 1, "numeric_range": [2.0, 2.0]}, "maximum_order": {"missing": 0, "unique_nonempty": 7, "numeric_range": [7.0, 13.0]}, "unit_benefit_cents": {"missing": 0, "unique_nonempty": 25, "numeric_range": [252.0, 9000.0]}, "item_fee_cents": {"missing": 0, "unique_nonempty": 28, "numeric_range": [101.0, 498.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "authorized", "matching_columns": [], "exact_matching_columns": 0}, {"term": "category", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item", "matching_columns": [{"column": "table", "exact": 28, "prefix": 28, "contains": 28, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "item_fee_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "maximum_order", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit_benefit_cents", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_04/export_04.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 84
Columns: ['table', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
table      item_ref resource amount  unit
usage Ra2a0eb0dfe79    power      6   kwh
usage Re0a6ccc90076    power      8   kwh
usage Rafc569d2d978    power      8   kwh
usage R640d4c97690d    space      5 liter
usage R5dfabf4dd5b4    space      5 liter
usage R025a74a1a060    power      4   kwh
usage R436178c5636f    labor      3  hour
usage R83c7f04f7b54    power      5   kwh
usage Rdce4df21cd05    labor      4  hour
usage Re3f595d2bf06    labor      5  hour
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 28}, "resource": {"missing": 0, "unique_nonempty": 3}, "amount": {"missing": 0, "unique_nonempty": 8, "numeric_range": [2.0, 9.0]}, "unit": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "hour", "matching_columns": [{"column": "unit", "exact": 28, "prefix": 28, "contains": 28, "examples": ["hour"]}], "exact_matching_columns": 1}, {"term": "kwh", "matching_columns": [{"column": "unit", "exact": 28, "prefix": 28, "contains": 28, "examples": ["kwh"]}], "exact_matching_columns": 1}, {"term": "liter", "matching_columns": [{"column": "unit", "exact": 28, "prefix": 28, "contains": 28, "examples": ["liter"]}], "exact_matching_columns": 1}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 84, "prefix": 84, "contains": 84, "examples": ["usage"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_05/export_05.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['table', 'category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'category': 'object', 'minimum_quantity': 'int64', 'maximum_quantity': 'int64', 'activation_fee_cents': 'int64'}
Preview only (first 10 rows):
   table category minimum_quantity maximum_quantity activation_fee_cents
category       G2                4               16                  150
category       G0                7               19                  277
category       G3               10               22                  187
category       G1                9               21                  343
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "category": {"missing": 0, "unique_nonempty": 4}, "minimum_quantity": {"missing": 0, "unique_nonempty": 4, "numeric_range": [4.0, 10.0]}, "maximum_quantity": {"missing": 0, "unique_nonempty": 4, "numeric_range": [16.0, 22.0]}, "activation_fee_cents": {"missing": 0, "unique_nonempty": 4, "numeric_range": [150.0, 343.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "activation_fee_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "category", "matching_columns": [{"column": "table", "exact": 4, "prefix": 4, "contains": 4, "examples": ["category"]}], "exact_matching_columns": 1}, {"term": "maximum_quantity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_quantity", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_06/export_06.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['table', 'item_a', 'item_b']
Parsed column types: {'table': 'object', 'item_a': 'object', 'item_b': 'object'}
Preview only (first 10 rows):
       table        item_a        item_b
incompatible R7e2a2c482e77 R189b5f29bad1
incompatible Rec957fff5de1 Rc3e636c6aeef
incompatible Rc3e636c6aeef Rdce4df21cd05
incompatible Rc4fc45bcfcbb R025a74a1a060
incompatible R436178c5636f R3ae1681715ca
incompatible Rc3e636c6aeef R802e24035203
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 5}, "item_b": {"missing": 0, "unique_nonempty": 6}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "incompatible", "matching_columns": [{"column": "table", "exact": 6, "prefix": 6, "contains": 6, "examples": ["incompatible"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_07.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 28
Columns: ['table', 'ref', 'kind', 'entity_id']
Parsed column types: {'table': 'object', 'ref': 'object', 'kind': 'object', 'entity_id': 'object'}
Preview only (first 10 rows):
   table           ref kind entity_id
identity R7e2a2c482e77 item       I27
identity R7e7dbfff0dc6 item       I03
identity Re3f595d2bf06 item       I26
identity R78298f0f56a2 item       I01
identity Rec957fff5de1 item       I09
identity Rafc569d2d978 item       I08
identity R436178c5636f item       I11
identity R0ae04f41cbb0 item       I18
identity R025a74a1a060 item       I25
identity R2efa1a0852bf item       I00
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "ref": {"missing": 0, "unique_nonempty": 28}, "kind": {"missing": 0, "unique_nonempty": 1}, "entity_id": {"missing": 0, "unique_nonempty": 28}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item", "matching_columns": [{"column": "kind", "exact": 28, "prefix": 28, "contains": 28, "examples": ["item"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_08.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['table', 'item_a', 'item_b', 'bonus_cents']
Parsed column types: {'table': 'object', 'item_a': 'object', 'item_b': 'object', 'bonus_cents': 'int64'}
Preview only (first 10 rows):
 table        item_a        item_b bonus_cents
bundle Re3f595d2bf06 R19b312900bfe         196
bundle Rafc569d2d978 R640d4c97690d         256
bundle Re3f595d2bf06 Re0a6ccc90076         153
bundle Rc3e636c6aeef Rafc569d2d978         227
bundle Re3f595d2bf06 R640d4c97690d         101
bundle Rc3e636c6aeef Re3f595d2bf06         394
bundle R83c7f04f7b54 R436178c5636f         454
bundle R3ae1681715ca R7e7dbfff0dc6         409
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 5}, "item_b": {"missing": 0, "unique_nonempty": 7}, "bonus_cents": {"missing": 0, "unique_nonempty": 8, "numeric_range": [101.0, 454.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "bonus_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "bundle", "matching_columns": [{"column": "table", "exact": 8, "prefix": 8, "contains": 8, "examples": ["bundle"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_09.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['table', 'resource', 'entry', 'amount', 'unit']
Parsed column types: {'table': 'object', 'resource': 'object', 'entry': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
          table resource       entry amount   unit
capacity_ledger    labor     opening  11820 minute
capacity_ledger    space     opening 276000     ml
capacity_ledger    labor reservation   -360 minute
capacity_ledger    power     opening 212000     wh
capacity_ledger    space reservation  -6000     ml
capacity_ledger    power reservation  -6000     wh
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "resource": {"missing": 0, "unique_nonempty": 3}, "entry": {"missing": 0, "unique_nonempty": 2}, "amount": {"missing": 0, "unique_nonempty": 5, "numeric_range": [-6000.0, 276000.0]}, "unit": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity_ledger", "matching_columns": [{"column": "table", "exact": 6, "prefix": 6, "contains": 6, "examples": ["capacity_ledger"]}], "exact_matching_columns": 1}, {"term": "ml", "matching_columns": [{"column": "unit", "exact": 2, "prefix": 2, "contains": 2, "examples": ["ml"]}], "exact_matching_columns": 1}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "wh", "matching_columns": [{"column": "unit", "exact": 2, "prefix": 2, "contains": 2, "examples": ["wh"]}], "exact_matching_columns": 1}]