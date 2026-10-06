File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_01/export_01.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['table', 'item_a', 'item_b', 'bonus_cents']
Parsed column types: {'table': 'object', 'item_a': 'object', 'item_b': 'object', 'bonus_cents': 'int64'}
Preview only (first 10 rows):
 table        item_a        item_b bonus_cents
bundle Rb3c64c386e4e Ra083d004fdf3         482
bundle Rc3f9ca031c8e R7452a15d3717         576
bundle Racf5f8a4fac2 Ref4615791cb9         502
bundle R7afbee5ee47b Rd501de06348a         186
bundle Rc3f9ca031c8e Ra04c52fef78e         127
bundle R7afbee5ee47b Rb3c64c386e4e         463
bundle R7fbb3730d314 Ref4615791cb9         374
bundle R5b0c501469fd R363320090a95         215
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 6}, "item_b": {"missing": 0, "unique_nonempty": 7}, "bonus_cents": {"missing": 0, "unique_nonempty": 8, "numeric_range": [127.0, 576.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "bonus_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "bundle", "matching_columns": [{"column": "table", "exact": 8, "prefix": 8, "contains": 8, "examples": ["bundle"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_02/export_02.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['table', 'resource', 'entry', 'amount', 'unit']
Parsed column types: {'table': 'object', 'resource': 'object', 'entry': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
          table resource       entry amount   unit
capacity_ledger    space reservation  -5000     ml
capacity_ledger    power     opening 212000     wh
capacity_ledger    labor     opening  15540 minute
capacity_ledger    labor reservation   -720 minute
capacity_ledger    space     opening 224000     ml
capacity_ledger    power reservation  -6000     wh
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "resource": {"missing": 0, "unique_nonempty": 3}, "entry": {"missing": 0, "unique_nonempty": 2}, "amount": {"missing": 0, "unique_nonempty": 6, "numeric_range": [-6000.0, 224000.0]}, "unit": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity_ledger", "matching_columns": [{"column": "table", "exact": 6, "prefix": 6, "contains": 6, "examples": ["capacity_ledger"]}], "exact_matching_columns": 1}, {"term": "ml", "matching_columns": [{"column": "unit", "exact": 2, "prefix": 2, "contains": 2, "examples": ["ml"]}], "exact_matching_columns": 1}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "wh", "matching_columns": [{"column": "unit", "exact": 2, "prefix": 2, "contains": 2, "examples": ["wh"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_03/export_03.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['table', 'category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'category': 'object', 'minimum_quantity': 'int64', 'maximum_quantity': 'int64', 'activation_fee_cents': 'int64'}
Preview only (first 10 rows):
   table category minimum_quantity maximum_quantity activation_fee_cents
category       G3                9               21                  193
category       G2                6               18                  324
category       G1                7               19                  263
category       G0                7               19                  140
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "category": {"missing": 0, "unique_nonempty": 4}, "minimum_quantity": {"missing": 0, "unique_nonempty": 3, "numeric_range": [6.0, 9.0]}, "maximum_quantity": {"missing": 0, "unique_nonempty": 3, "numeric_range": [18.0, 21.0]}, "activation_fee_cents": {"missing": 0, "unique_nonempty": 4, "numeric_range": [140.0, 324.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "activation_fee_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "category", "matching_columns": [{"column": "table", "exact": 4, "prefix": 4, "contains": 4, "examples": ["category"]}], "exact_matching_columns": 1}, {"term": "maximum_quantity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_quantity", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_04/export_04.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 24
Columns: ['table', 'ref', 'kind', 'entity_id']
Parsed column types: {'table': 'object', 'ref': 'object', 'kind': 'object', 'entity_id': 'object'}
Preview only (first 10 rows):
   table           ref kind entity_id
identity Ra04c52fef78e item       I08
identity Ra083d004fdf3 item       I00
identity R7afbee5ee47b item       I03
identity Rb4a56a392c2a item       I17
identity Racf5f8a4fac2 item       I04
identity Rc3f9ca031c8e item       I16
identity Rd501de06348a item       I11
identity R887ffaea87dd item       I23
identity R634e3af40d39 item       I15
identity R7452a15d3717 item       I05
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "ref": {"missing": 0, "unique_nonempty": 24}, "kind": {"missing": 0, "unique_nonempty": 1}, "entity_id": {"missing": 0, "unique_nonempty": 24}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item", "matching_columns": [{"column": "kind", "exact": 24, "prefix": 24, "contains": 24, "examples": ["item"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_05/export_05.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['table', 'item_a', 'item_b']
Parsed column types: {'table': 'object', 'item_a': 'object', 'item_b': 'object'}
Preview only (first 10 rows):
       table        item_a        item_b
incompatible Rb4a56a392c2a R7afbee5ee47b
incompatible Rd501de06348a Rb4a56a392c2a
incompatible R6cdd375c7a2a R7fbb3730d314
incompatible R887ffaea87dd R6cdd375c7a2a
incompatible R828da38f58bc R887ffaea87dd
incompatible Ref4615791cb9 R7fbb3730d314
incompatible R35f4ccfb1574 R6cdd375c7a2a
incompatible R863022fbf3f2 R7fbb3730d314
incompatible R7fbb3730d314 R828da38f58bc
incompatible R828da38f58bc R6a8f63724def
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 9}, "item_b": {"missing": 0, "unique_nonempty": 7}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "incompatible", "matching_columns": [{"column": "table", "exact": 10, "prefix": 10, "contains": 10, "examples": ["incompatible"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_06/export_06.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 24
Columns: ['table', 'item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'unit_benefit_cents', 'item_fee_cents']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'category': 'object', 'authorized': 'int64', 'minimum_lot': 'int64', 'maximum_order': 'int64', 'unit_benefit_cents': 'int64', 'item_fee_cents': 'int64'}
Preview only (first 10 rows):
table      item_ref category authorized minimum_lot maximum_order unit_benefit_cents item_fee_cents
 item R363320090a95       G1          1           2             9                338            431
 item R798a1de24f1f       G2          1           2             7                482            155
 item Rb3c64c386e4e       G2          1           2            10                369            301
 item Ra083d004fdf3       G0          1           2             8                524            432
 item R7fbb3730d314       G1          0           2             7               9000            428
 item Rd501de06348a       G3          1           2            12                491            258
 item Racf5f8a4fac2       G0          1           2            11                581            240
 item Re4688318762d       G3          1           2            13                887            127
 item Ra032aa43e993       G3          1           2             7               1425            184
 item R6cdd375c7a2a       G0          1           2             8               1551            445
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 24}, "category": {"missing": 0, "unique_nonempty": 4}, "authorized": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}, "minimum_lot": {"missing": 0, "unique_nonempty": 1, "numeric_range": [2.0, 2.0]}, "maximum_order": {"missing": 0, "unique_nonempty": 7, "numeric_range": [7.0, 13.0]}, "unit_benefit_cents": {"missing": 0, "unique_nonempty": 21, "numeric_range": [316.0, 9000.0]}, "item_fee_cents": {"missing": 0, "unique_nonempty": 23, "numeric_range": [116.0, 445.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "authorized", "matching_columns": [], "exact_matching_columns": 0}, {"term": "category", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item", "matching_columns": [{"column": "table", "exact": 24, "prefix": 24, "contains": 24, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "item_fee_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "maximum_order", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit_benefit_cents", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_01/export_07.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 24
Columns: ['table', 'item_ref', 'historical_sales', 'forecast_units', 'margin_percent']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'historical_sales': 'int64', 'forecast_units': 'int64', 'margin_percent': 'int64'}
Preview only (first 10 rows):
 table      item_ref historical_sales forecast_units margin_percent
market Re4688318762d                3              1             33
market R798a1de24f1f                2              3             21
market Ra04c52fef78e                1              2             22
market R7afbee5ee47b                2              1             27
market R7fbb3730d314                2              3             31
market R35f4ccfb1574                2              1             15
market R828da38f58bc                1              3             15
market Rc3f9ca031c8e                2              2             16
market Ra083d004fdf3                1              3             21
market Rd501de06348a                1              1             15
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 24}, "historical_sales": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "forecast_units": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "margin_percent": {"missing": 0, "unique_nonempty": 16, "numeric_range": [7.0, 34.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: []

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_02/export_08.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 12
Columns: ['table', 'item_ref', 'prerequisite_ref']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'prerequisite_ref': 'object'}
Preview only (first 10 rows):
   table      item_ref prerequisite_ref
requires R35f4ccfb1574    Ra04c52fef78e
requires Ra032aa43e993    Racf5f8a4fac2
requires R887ffaea87dd    Ra04c52fef78e
requires Rc3f9ca031c8e    Re4688318762d
requires Rbe86b9a56e99    Ra083d004fdf3
requires R6cdd375c7a2a    Racf5f8a4fac2
requires Rb4a56a392c2a    R828da38f58bc
requires R634e3af40d39    Ra083d004fdf3
requires R798a1de24f1f    Racf5f8a4fac2
requires Ref4615791cb9    Rb3c64c386e4e
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 12}, "prerequisite_ref": {"missing": 0, "unique_nonempty": 8}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "requires", "matching_columns": [{"column": "table", "exact": 12, "prefix": 12, "contains": 12, "examples": ["requires"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_03/export_09.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 72
Columns: ['table', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
table      item_ref resource amount  unit
usage Rb4a56a392c2a    power      4   kwh
usage Ra04c52fef78e    space      3 liter
usage Ra083d004fdf3    space      9 liter
usage Racf5f8a4fac2    power      2   kwh
usage R798a1de24f1f    labor      8  hour
usage R798a1de24f1f    space      8 liter
usage R35f4ccfb1574    labor      6  hour
usage R363320090a95    labor      4  hour
usage R863022fbf3f2    labor      4  hour
usage R828da38f58bc    space      2 liter
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 24}, "resource": {"missing": 0, "unique_nonempty": 3}, "amount": {"missing": 0, "unique_nonempty": 8, "numeric_range": [2.0, 9.0]}, "unit": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "hour", "matching_columns": [{"column": "unit", "exact": 24, "prefix": 24, "contains": 24, "examples": ["hour"]}], "exact_matching_columns": 1}, {"term": "kwh", "matching_columns": [{"column": "unit", "exact": 24, "prefix": 24, "contains": 24, "examples": ["kwh"]}], "exact_matching_columns": 1}, {"term": "liter", "matching_columns": [{"column": "unit", "exact": 24, "prefix": 24, "contains": 24, "examples": ["liter"]}], "exact_matching_columns": 1}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 72, "prefix": 72, "contains": 72, "examples": ["usage"]}], "exact_matching_columns": 1}]