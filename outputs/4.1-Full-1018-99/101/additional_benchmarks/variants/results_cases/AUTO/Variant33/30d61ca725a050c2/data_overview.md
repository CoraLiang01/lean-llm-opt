File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_01.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 48
Columns: ['table', 'item_ref', 'component', 'amount_cents']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'component': 'object', 'amount_cents': 'int64'}
Preview only (first 10 rows):
  table      item_ref     component amount_cents
benefit R5ad6394143cf   item12_base         9043
benefit Rd791f4961a31 item10_rebate           -8
benefit R42e7427144fb    item0_base          437
benefit R7c46b20b51e0   item21_base          889
benefit Rd02fcc260ac2   item18_base         9027
benefit R3e7214cc1c8d   item15_base         1314
benefit R83c306294c58 item22_rebate          -23
benefit R604e812d1798    item9_base         1126
benefit R5948d9dfbde3 item13_rebate           -4
benefit R11f2f5eb23eb  item7_rebate          -28
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 24}, "component": {"missing": 0, "unique_nonempty": 48}, "amount_cents": {"missing": 0, "unique_nonempty": 41, "numeric_range": [-47.0, 9045.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "benefit", "matching_columns": [{"column": "table", "exact": 48, "prefix": 48, "contains": 48, "examples": ["benefit"]}], "exact_matching_columns": 1}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_02.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['table', 'item_a', 'item_b', 'bonus_cents']
Parsed column types: {'table': 'object', 'item_a': 'object', 'item_b': 'object', 'bonus_cents': 'int64'}
Preview only (first 10 rows):
 table        item_a        item_b bonus_cents
bundle R604e812d1798 Rd02fcc260ac2         181
bundle R83c306294c58 R5ad6394143cf         398
bundle Rfc4632d553a5 R3e7214cc1c8d         352
bundle Rcb3019602c46 Rd99d7cf392a6         425
bundle Rc078fedbc2c8 R64096cbb3924         328
bundle R5cef9f0f8c01 Rd99d7cf392a6         274
bundle R40412b996535 R5cef9f0f8c01         383
bundle R42e7427144fb R5cef9f0f8c01         218
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 8}, "item_b": {"missing": 0, "unique_nonempty": 6}, "bonus_cents": {"missing": 0, "unique_nonempty": 8, "numeric_range": [181.0, 425.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "bonus_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "bundle", "matching_columns": [{"column": "table", "exact": 8, "prefix": 8, "contains": 8, "examples": ["bundle"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_03.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['table', 'resource', 'entry', 'amount', 'unit']
Parsed column types: {'table': 'object', 'resource': 'object', 'entry': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
          table resource       entry amount   unit
capacity_ledger    labor     opening  12480 minute
capacity_ledger    space     opening 310000     ml
capacity_ledger    power     opening 233000     wh
capacity_ledger    space reservation  -6000     ml
capacity_ledger    power reservation -10000     wh
capacity_ledger    labor reservation   -300 minute
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "resource": {"missing": 0, "unique_nonempty": 3}, "entry": {"missing": 0, "unique_nonempty": 2}, "amount": {"missing": 0, "unique_nonempty": 6, "numeric_range": [-10000.0, 310000.0]}, "unit": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity_ledger", "matching_columns": [{"column": "table", "exact": 6, "prefix": 6, "contains": 6, "examples": ["capacity_ledger"]}], "exact_matching_columns": 1}, {"term": "labor", "matching_columns": [{"column": "resource", "exact": 2, "prefix": 2, "contains": 2, "examples": ["labor"]}], "exact_matching_columns": 1}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_04.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['table', 'category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'category': 'object', 'minimum_quantity': 'int64', 'maximum_quantity': 'int64', 'activation_fee_cents': 'int64'}
Preview only (first 10 rows):
   table category minimum_quantity maximum_quantity activation_fee_cents
category       G1                5               17                  190
category       G3                9               21                  215
category       G2                8               20                   98
category       G0                9               21                  120
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "category": {"missing": 0, "unique_nonempty": 4}, "minimum_quantity": {"missing": 0, "unique_nonempty": 3, "numeric_range": [5.0, 9.0]}, "maximum_quantity": {"missing": 0, "unique_nonempty": 3, "numeric_range": [17.0, 21.0]}, "activation_fee_cents": {"missing": 0, "unique_nonempty": 4, "numeric_range": [98.0, 215.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "activation_fee_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "category", "matching_columns": [{"column": "table", "exact": 4, "prefix": 4, "contains": 4, "examples": ["category"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_05.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 24
Columns: ['table', 'ref', 'kind', 'entity_id', 'display_name']
Parsed column types: {'table': 'object', 'ref': 'object', 'kind': 'object', 'entity_id': 'object', 'display_name': 'object'}
Preview only (first 10 rows):
   table           ref kind     entity_id    display_name
identity R3098f5b30478 item RA7_OPTION_21          Queens
identity R64096cbb3924 item RA7_OPTION_05   Staten Island
identity R5ad6394143cf item RA7_OPTION_13        Flatbush
identity R3b6c3396ec6d item RA7_OPTION_15      Park Slope
identity R5cef9f0f8c01 item RA7_OPTION_09         Midtown
identity Rd791f4961a31 item RA7_OPTION_11    Williamsburg
identity R671e240d4376 item RA7_OPTION_17 Jackson Heights
identity R83c306294c58 item RA7_OPTION_23       Manhattan
identity R46f743c54f37 item RA7_OPTION_07 Upper East Side
identity R42e7427144fb item RA7_OPTION_01          Queens
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "ref": {"missing": 0, "unique_nonempty": 24}, "kind": {"missing": 0, "unique_nonempty": 1}, "entity_id": {"missing": 0, "unique_nonempty": 24}, "display_name": {"missing": 0, "unique_nonempty": 20}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item", "matching_columns": [{"column": "kind", "exact": 24, "prefix": 24, "contains": 24, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_06.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['table', 'item_a', 'item_b']
Parsed column types: {'table': 'object', 'item_a': 'object', 'item_b': 'object'}
Preview only (first 10 rows):
       table        item_a        item_b
incompatible Rfc4632d553a5 R3e7214cc1c8d
incompatible R24e2cfc15d68 Rf901d8dca0b2
incompatible Rcb3019602c46 R42e7427144fb
incompatible R671e240d4376 R5cef9f0f8c01
incompatible Rf901d8dca0b2 R604e812d1798
incompatible R64096cbb3924 R5948d9dfbde3
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 6}, "item_b": {"missing": 0, "unique_nonempty": 6}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "incompatible", "matching_columns": [{"column": "table", "exact": 6, "prefix": 6, "contains": 6, "examples": ["incompatible"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_07.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 12
Columns: ['table', 'item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'configuration_id']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'category': 'object', 'authorized': 'int64', 'minimum_lot': 'int64', 'maximum_order': 'int64', 'configuration_id': 'object'}
Preview only (first 10 rows):
table      item_ref category authorized minimum_lot maximum_order configuration_id
 item R3098f5b30478       G0          1           2             8           CFG_02
 item R64096cbb3924       G0          1           2            11           CFG_01
 item R3b6c3396ec6d       G2          1           2             8           CFG_01
 item Rfc4632d553a5       G2          1           2            11           CFG_01
 item R671e240d4376       G0          0           2            11           CFG_01
 item R46f743c54f37       G2          1           2            11           CFG_01
 item R604e812d1798       G1          1           2             7           CFG_01
 item Rc078fedbc2c8       G1          1           2            13           CFG_01
 item Rd99d7cf392a6       G3          1           2             8           CFG_01
 item R7c46b20b51e0       G1          1           2            10           CFG_02
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 12}, "category": {"missing": 0, "unique_nonempty": 4}, "authorized": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}, "minimum_lot": {"missing": 0, "unique_nonempty": 1, "numeric_range": [2.0, 2.0]}, "maximum_order": {"missing": 0, "unique_nonempty": 6, "numeric_range": [7.0, 13.0]}, "configuration_id": {"missing": 0, "unique_nonempty": 2}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "category", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item", "matching_columns": [{"column": "table", "exact": 12, "prefix": 12, "contains": 12, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "maximum_order", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_08.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 12
Columns: ['table', 'item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'configuration_id']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'category': 'object', 'authorized': 'int64', 'minimum_lot': 'int64', 'maximum_order': 'int64', 'configuration_id': 'object'}
Preview only (first 10 rows):
table      item_ref category authorized minimum_lot maximum_order configuration_id
 item Rd02fcc260ac2       G2          0           2             9           CFG_01
 item Rd791f4961a31       G2          1           2            11           CFG_01
 item R83c306294c58       G2          1           2             7           CFG_02
 item R42e7427144fb       G0          1           2            12           CFG_01
 item R5cef9f0f8c01       G0          1           2            10           CFG_01
 item R5ad6394143cf       G0          0           2            11           CFG_01
 item R11f2f5eb23eb       G3          1           2             9           CFG_01
 item R40412b996535       G3          1           2            10           CFG_01
 item R24e2cfc15d68       G1          1           2             9           CFG_01
 item R5948d9dfbde3       G1          1           2            10           CFG_01
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 12}, "category": {"missing": 0, "unique_nonempty": 4}, "authorized": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}, "minimum_lot": {"missing": 0, "unique_nonempty": 1, "numeric_range": [2.0, 2.0]}, "maximum_order": {"missing": 0, "unique_nonempty": 6, "numeric_range": [7.0, 12.0]}, "configuration_id": {"missing": 0, "unique_nonempty": 2}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "category", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item", "matching_columns": [{"column": "table", "exact": 12, "prefix": 12, "contains": 12, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "maximum_order", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_09.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 24
Columns: ['table', 'item_ref', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'activation_fee_cents': 'int64'}
Preview only (first 10 rows):
   table      item_ref activation_fee_cents
item_fee R64096cbb3924                  311
item_fee Rd02fcc260ac2                  347
item_fee R5ad6394143cf                  326
item_fee R42e7427144fb                  392
item_fee R671e240d4376                  484
item_fee Rd791f4961a31                  349
item_fee R3b6c3396ec6d                  247
item_fee R83c306294c58                  426
item_fee R3098f5b30478                  271
item_fee R46f743c54f37                  382
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 24}, "activation_fee_cents": {"missing": 0, "unique_nonempty": 24, "numeric_range": [172.0, 492.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "activation_fee_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item_fee", "matching_columns": [{"column": "table", "exact": 24, "prefix": 24, "contains": 24, "examples": ["item_fee"]}], "exact_matching_columns": 1}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_10.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 24
Columns: ['table', 'item_ref', 'historical_sales', 'forecast_units', 'margin_percent']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'historical_sales': 'int64', 'forecast_units': 'int64', 'margin_percent': 'int64'}
Preview only (first 10 rows):
 table      item_ref historical_sales forecast_units margin_percent
market R3b6c3396ec6d                2              1             20
market Rfc4632d553a5                3              3             15
market R3098f5b30478                3              3             17
market Rd02fcc260ac2                1              1             35
market R5cef9f0f8c01                2              2             14
market R5ad6394143cf                3              2             11
market R42e7427144fb                3              1              9
market R83c306294c58                2              1             35
market R46f743c54f37                1              3             30
market R671e240d4376                2              1             32
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 24}, "historical_sales": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "forecast_units": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "margin_percent": {"missing": 0, "unique_nonempty": 16, "numeric_range": [6.0, 35.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_11.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 12
Columns: ['table', 'item_ref', 'prerequisite_ref']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'prerequisite_ref': 'object'}
Preview only (first 10 rows):
   table      item_ref prerequisite_ref
requires R3e7214cc1c8d    Rd791f4961a31
requires Rdc058ac3690a    R604e812d1798
requires R5948d9dfbde3    R5cef9f0f8c01
requires Rf901d8dca0b2    R46f743c54f37
requires Rcb3019602c46    R64096cbb3924
requires R7c46b20b51e0    R64096cbb3924
requires R3098f5b30478    Rd99d7cf392a6
requires R671e240d4376    R5cef9f0f8c01
requires R83c306294c58    R5cef9f0f8c01
requires R5ad6394143cf    R64096cbb3924
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 12}, "prerequisite_ref": {"missing": 0, "unique_nonempty": 7}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "prerequisite_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "requires", "matching_columns": [{"column": "table", "exact": 12, "prefix": 12, "contains": 12, "examples": ["requires"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_12.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 36
Columns: ['table', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
table      item_ref resource amount unit
usage Rd02fcc260ac2    space   3000   ml
usage R24e2cfc15d68    space   5000   ml
usage R46f743c54f37    space   6000   ml
usage R5948d9dfbde3    space   3000   ml
usage R5ad6394143cf    space   9000   ml
usage Rd791f4961a31    space   6000   ml
usage Rd99d7cf392a6    space   9000   ml
usage R5cef9f0f8c01    space   8000   ml
usage R3e7214cc1c8d    space   2000   ml
usage R42e7427144fb    space   8000   ml
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 24}, "resource": {"missing": 0, "unique_nonempty": 3}, "amount": {"missing": 0, "unique_nonempty": 14, "numeric_range": [120.0, 9000.0]}, "unit": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "labor", "matching_columns": [{"column": "resource", "exact": 12, "prefix": 12, "contains": 12, "examples": ["labor"]}], "exact_matching_columns": 1}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 36, "prefix": 36, "contains": 36, "examples": ["usage"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_13.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 36
Columns: ['table', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
table      item_ref resource amount unit
usage R11f2f5eb23eb    space   7000   ml
usage R40412b996535    space   2000   ml
usage R3b6c3396ec6d    space   5000   ml
usage Rc078fedbc2c8    space   7000   ml
usage R83c306294c58    space   8000   ml
usage R671e240d4376    space   8000   ml
usage Rfc4632d553a5    space   7000   ml
usage R7c46b20b51e0    space   6000   ml
usage Rf901d8dca0b2    space   9000   ml
usage Rdc058ac3690a    space   8000   ml
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 22}, "resource": {"missing": 0, "unique_nonempty": 3}, "amount": {"missing": 0, "unique_nonempty": 14, "numeric_range": [120.0, 9000.0]}, "unit": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "labor", "matching_columns": [{"column": "resource", "exact": 12, "prefix": 12, "contains": 12, "examples": ["labor"]}], "exact_matching_columns": 1}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 36, "prefix": 36, "contains": 36, "examples": ["usage"]}], "exact_matching_columns": 1}]