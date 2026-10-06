File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_01/export_01.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 60
Columns: ['table', 'item_ref', 'component', 'amount_cents']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'component': 'object', 'amount_cents': 'int64'}
Preview only (first 10 rows):
  table      item_ref     component amount_cents
benefit Raf6249743aab   item24_base          674
benefit R77b5ceefbc30 item22_rebate          -24
benefit R1f5cabe17f56   item12_base         1051
benefit R51e14ed234fb item25_rebate          -18
benefit Ra54c415fd225  item7_rebate          -19
benefit Rdb9e5c3398ec    item3_base         1283
benefit R6ff46dde76b5 item19_rebate          -34
benefit R21d9ec858608   item27_base         9044
benefit R5e5bafcd3a11    item9_base          490
benefit R68d1acbf1b45  item1_rebate          -15
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 30}, "component": {"missing": 0, "unique_nonempty": 60}, "amount_cents": {"missing": 0, "unique_nonempty": 50, "numeric_range": [-49.0, 9044.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "benefit", "matching_columns": [{"column": "table", "exact": 60, "prefix": 60, "contains": 60, "examples": ["benefit"]}], "exact_matching_columns": 1}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_02/export_02.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['table', 'item_a', 'item_b', 'bonus_cents']
Parsed column types: {'table': 'object', 'item_a': 'object', 'item_b': 'object', 'bonus_cents': 'int64'}
Preview only (first 10 rows):
 table        item_a        item_b bonus_cents
bundle R911cbab9a301 R81173150688b         369
bundle Rd654f83d9c9b R51e14ed234fb         425
bundle Rcf72b0319ae6 R18fe9c1e5f4c         228
bundle Rcf2a2fbd9a16 R6ff46dde76b5         241
bundle R6b719309ca14 R23ae26e45153         215
bundle R144b336321b3 R23ae26e45153         431
bundle R6b719309ca14 Raf6249743aab         158
bundle Raf6249743aab R144b336321b3         352
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 7}, "item_b": {"missing": 0, "unique_nonempty": 7}, "bonus_cents": {"missing": 0, "unique_nonempty": 8, "numeric_range": [158.0, 431.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "bonus_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "bundle", "matching_columns": [{"column": "table", "exact": 8, "prefix": 8, "contains": 8, "examples": ["bundle"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_03/export_03.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['table', 'resource', 'entry', 'amount', 'unit']
Parsed column types: {'table': 'object', 'resource': 'object', 'entry': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
          table  resource       entry amount unit
capacity_ledger SECTION_C     opening 227000   ml
capacity_ledger SECTION_A     opening 234000   ml
capacity_ledger SECTION_B     opening 267000   ml
capacity_ledger SECTION_B reservation  -9000   ml
capacity_ledger SECTION_C reservation -11000   ml
capacity_ledger SECTION_A reservation -11000   ml
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "resource": {"missing": 0, "unique_nonempty": 3}, "entry": {"missing": 0, "unique_nonempty": 2}, "amount": {"missing": 0, "unique_nonempty": 5, "numeric_range": [-11000.0, 267000.0]}, "unit": {"missing": 0, "unique_nonempty": 1}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity_ledger", "matching_columns": [{"column": "table", "exact": 6, "prefix": 6, "contains": 6, "examples": ["capacity_ledger"]}], "exact_matching_columns": 1}, {"term": "ml", "matching_columns": [{"column": "unit", "exact": 6, "prefix": 6, "contains": 6, "examples": ["ml"]}], "exact_matching_columns": 1}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_04/export_04.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['table', 'category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'category': 'object', 'minimum_quantity': 'int64', 'maximum_quantity': 'int64', 'activation_fee_cents': 'int64'}
Preview only (first 10 rows):
   table category minimum_quantity maximum_quantity activation_fee_cents
category       G2                7               19                  215
category       G0                7               19                  180
category       G1                8               20                  382
category       G3               10               22                  164
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "category": {"missing": 0, "unique_nonempty": 4}, "minimum_quantity": {"missing": 0, "unique_nonempty": 3, "numeric_range": [7.0, 10.0]}, "maximum_quantity": {"missing": 0, "unique_nonempty": 3, "numeric_range": [19.0, 22.0]}, "activation_fee_cents": {"missing": 0, "unique_nonempty": 4, "numeric_range": [164.0, 382.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "activation_fee_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "category", "matching_columns": [{"column": "table", "exact": 4, "prefix": 4, "contains": 4, "examples": ["category"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_05/export_05.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 30
Columns: ['table', 'ref', 'kind', 'entity_id', 'display_name']
Parsed column types: {'table': 'object', 'ref': 'object', 'kind': 'object', 'entity_id': 'object', 'display_name': 'int64'}
Preview only (first 10 rows):
   table           ref kind      entity_id display_name
identity R6b719309ca14 item RA10_OPTION_11            1
identity Raf6249743aab item RA10_OPTION_25            5
identity Rcf2a2fbd9a16 item RA10_OPTION_21            1
identity R1f5cabe17f56 item RA10_OPTION_13            3
identity R18fe9c1e5f4c item RA10_OPTION_15            5
identity R3f4281b8fad6 item RA10_OPTION_27            7
identity R35cb0535ec96 item RA10_OPTION_05            5
identity Rcf72b0319ae6 item RA10_OPTION_17            7
identity R86c8af615f03 item RA10_OPTION_03            3
identity Rbe862dd546a3 item RA10_OPTION_01            1
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "ref": {"missing": 0, "unique_nonempty": 30}, "kind": {"missing": 0, "unique_nonempty": 1}, "entity_id": {"missing": 0, "unique_nonempty": 30}, "display_name": {"missing": 0, "unique_nonempty": 10, "numeric_range": [1.0, 10.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item", "matching_columns": [{"column": "kind", "exact": 30, "prefix": 30, "contains": 30, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_06/export_06.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['table', 'item_a', 'item_b']
Parsed column types: {'table': 'object', 'item_a': 'object', 'item_b': 'object'}
Preview only (first 10 rows):
       table        item_a        item_b
incompatible Re218e287b09c Rbadcf0b5397e
incompatible R694597094779 R18fe9c1e5f4c
incompatible R911cbab9a301 R5e5bafcd3a11
incompatible Raf6249743aab Rdb9e5c3398ec
incompatible R68d1acbf1b45 Rcf2a2fbd9a16
incompatible R1f5cabe17f56 R23ae26e45153
incompatible R1f5cabe17f56 R694597094779
incompatible R86c8af615f03 Raf6249743aab
incompatible Ra54c415fd225 R694597094779
incompatible R5e5bafcd3a11 Raf6249743aab
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 9}, "item_b": {"missing": 0, "unique_nonempty": 8}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "incompatible", "matching_columns": [{"column": "table", "exact": 10, "prefix": 10, "contains": 10, "examples": ["incompatible"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_01/export_07.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 15
Columns: ['table', 'item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'configuration_id', 'location_id']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'category': 'object', 'authorized': 'int64', 'minimum_lot': 'int64', 'maximum_order': 'int64', 'configuration_id': 'object', 'location_id': 'object'}
Preview only (first 10 rows):
table      item_ref category authorized minimum_lot maximum_order configuration_id location_id
 item R18fe9c1e5f4c       G2          1           2            13           CFG_02   SECTION_C
 item Rcf72b0319ae6       G0          1           2             9           CFG_02   SECTION_B
 item Rd654f83d9c9b       G2          1           2             8           CFG_01   SECTION_A
 item R77b5ceefbc30       G2          0           2            10           CFG_03   SECTION_B
 item R35cb0535ec96       G0          1           2            10           CFG_01   SECTION_B
 item R6b719309ca14       G2          1           2            13           CFG_02   SECTION_B
 item R86c8af615f03       G2          1           2            12           CFG_01   SECTION_C
 item R81173150688b       G0          1           2             9           CFG_01   SECTION_C
 item Rbadcf0b5397e       G3          1           2             7           CFG_03   SECTION_C
 item R6ff46dde76b5       G3          1           2             8           CFG_02   SECTION_B
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 15}, "category": {"missing": 0, "unique_nonempty": 4}, "authorized": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}, "minimum_lot": {"missing": 0, "unique_nonempty": 1, "numeric_range": [2.0, 2.0]}, "maximum_order": {"missing": 0, "unique_nonempty": 6, "numeric_range": [7.0, 13.0]}, "configuration_id": {"missing": 0, "unique_nonempty": 3}, "location_id": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "category", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item", "matching_columns": [{"column": "table", "exact": 15, "prefix": 15, "contains": 15, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "location_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "maximum_order", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_02/export_08.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 15
Columns: ['table', 'item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'configuration_id', 'location_id']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'category': 'object', 'authorized': 'int64', 'minimum_lot': 'int64', 'maximum_order': 'int64', 'configuration_id': 'object', 'location_id': 'object'}
Preview only (first 10 rows):
table      item_ref category authorized minimum_lot maximum_order configuration_id location_id
 item R1f5cabe17f56       G0          1           2            10           CFG_02   SECTION_A
 item R3f4281b8fad6       G2          0           2            10           CFG_03   SECTION_C
 item R8af59fb1b6d9       G2          1           2            10           CFG_02   SECTION_A
 item R144b336321b3       G0          0           2            13           CFG_03   SECTION_B
 item Raf6249743aab       G0          1           2             8           CFG_03   SECTION_A
 item Rcf2a2fbd9a16       G0          1           2             8           CFG_03   SECTION_C
 item Rbe862dd546a3       G0          1           2            11           CFG_01   SECTION_A
 item R694597094779       G1          1           2            10           CFG_02   SECTION_B
 item R672fb805523e       G1          1           2            13           CFG_02   SECTION_C
 item R5e5bafcd3a11       G1          1           2             7           CFG_01   SECTION_A
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 15}, "category": {"missing": 0, "unique_nonempty": 4}, "authorized": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}, "minimum_lot": {"missing": 0, "unique_nonempty": 1, "numeric_range": [2.0, 2.0]}, "maximum_order": {"missing": 0, "unique_nonempty": 5, "numeric_range": [7.0, 13.0]}, "configuration_id": {"missing": 0, "unique_nonempty": 3}, "location_id": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "category", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item", "matching_columns": [{"column": "table", "exact": 15, "prefix": 15, "contains": 15, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "location_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "maximum_order", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_03/export_09.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 30
Columns: ['table', 'item_ref', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'activation_fee_cents': 'int64'}
Preview only (first 10 rows):
   table      item_ref activation_fee_cents
item_fee R8af59fb1b6d9                  275
item_fee Rbe862dd546a3                  367
item_fee R18fe9c1e5f4c                  260
item_fee R77b5ceefbc30                  267
item_fee R81173150688b                  384
item_fee Raf6249743aab                  255
item_fee R1f5cabe17f56                  359
item_fee R86c8af615f03                  135
item_fee Rcf72b0319ae6                  354
item_fee R3f4281b8fad6                  314
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 30}, "activation_fee_cents": {"missing": 0, "unique_nonempty": 30, "numeric_range": [131.0, 482.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "activation_fee_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item_fee", "matching_columns": [{"column": "table", "exact": 30, "prefix": 30, "contains": 30, "examples": ["item_fee"]}], "exact_matching_columns": 1}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_04/export_10.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 30
Columns: ['table', 'item_ref', 'historical_sales', 'forecast_units', 'margin_percent']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'historical_sales': 'int64', 'forecast_units': 'int64', 'margin_percent': 'int64'}
Preview only (first 10 rows):
 table      item_ref historical_sales forecast_units margin_percent
market R18fe9c1e5f4c                3              3             35
market Rd654f83d9c9b                3              2             32
market R1f5cabe17f56                3              1             13
market R35cb0535ec96                2              1             19
market R77b5ceefbc30                2              2             11
market R144b336321b3                1              2             26
market R3f4281b8fad6                3              1             28
market Raf6249743aab                3              3             12
market R86c8af615f03                3              1             16
market Rbe862dd546a3                2              3             24
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 30}, "historical_sales": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "forecast_units": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "margin_percent": {"missing": 0, "unique_nonempty": 17, "numeric_range": [5.0, 35.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_05/export_11.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 18
Columns: ['table', 'item_ref', 'prerequisite_ref']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'prerequisite_ref': 'object'}
Preview only (first 10 rows):
   table      item_ref prerequisite_ref
requires Rcf72b0319ae6    R68d1acbf1b45
requires R3f4281b8fad6    R81173150688b
requires Rcf2a2fbd9a16    Re218e287b09c
requires R8af59fb1b6d9    Rdb9e5c3398ec
requires R18fe9c1e5f4c    R9cb1d79c603b
requires Raf6249743aab    R68d1acbf1b45
requires R1f5cabe17f56    Rd654f83d9c9b
requires R77b5ceefbc30    R6b719309ca14
requires R144b336321b3    R86c8af615f03
requires R6ff46dde76b5    Rbe862dd546a3
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 18}, "prerequisite_ref": {"missing": 0, "unique_nonempty": 11}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "prerequisite_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "requires", "matching_columns": [{"column": "table", "exact": 18, "prefix": 18, "contains": 18, "examples": ["requires"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_06/export_12.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 15
Columns: ['table', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
table      item_ref  resource amount  unit
usage R981c210270a2 SECTION_A      8 liter
usage Rbe862dd546a3 SECTION_A      3 liter
usage R21d9ec858608 SECTION_A      3 liter
usage R5e5bafcd3a11 SECTION_A      7 liter
usage Rdb9e5c3398ec SECTION_A      4 liter
usage Rcf72b0319ae6 SECTION_B      6 liter
usage Ra54c415fd225 SECTION_B      8 liter
usage R6ff46dde76b5 SECTION_B      7 liter
usage R35cb0535ec96 SECTION_B      8 liter
usage R68d1acbf1b45 SECTION_B      4 liter
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 15}, "resource": {"missing": 0, "unique_nonempty": 3}, "amount": {"missing": 0, "unique_nonempty": 6, "numeric_range": [2.0, 8.0]}, "unit": {"missing": 0, "unique_nonempty": 1}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "liter", "matching_columns": [{"column": "unit", "exact": 15, "prefix": 15, "contains": 15, "examples": ["liter"]}], "exact_matching_columns": 1}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 15, "prefix": 15, "contains": 15, "examples": ["usage"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_01/export_13.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 15
Columns: ['table', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
table      item_ref  resource amount  unit
usage R8af59fb1b6d9 SECTION_A      7 liter
usage Raf6249743aab SECTION_A      3 liter
usage R1f5cabe17f56 SECTION_A      7 liter
usage Rd654f83d9c9b SECTION_A      6 liter
usage R23ae26e45153 SECTION_A      3 liter
usage R6b719309ca14 SECTION_B      2 liter
usage R694597094779 SECTION_B      5 liter
usage R51e14ed234fb SECTION_B      8 liter
usage R144b336321b3 SECTION_B      6 liter
usage R77b5ceefbc30 SECTION_B      3 liter
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 15}, "resource": {"missing": 0, "unique_nonempty": 3}, "amount": {"missing": 0, "unique_nonempty": 7, "numeric_range": [2.0, 8.0]}, "unit": {"missing": 0, "unique_nonempty": 1}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "liter", "matching_columns": [{"column": "unit", "exact": 15, "prefix": 15, "contains": 15, "examples": ["liter"]}], "exact_matching_columns": 1}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 15, "prefix": 15, "contains": 15, "examples": ["usage"]}], "exact_matching_columns": 1}]