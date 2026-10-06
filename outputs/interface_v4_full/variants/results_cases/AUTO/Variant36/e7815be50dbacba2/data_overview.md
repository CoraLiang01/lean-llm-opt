File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_01/export_01.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['table', 'item_a', 'item_b', 'bonus_cents']
Parsed column types: {'table': 'object', 'item_a': 'object', 'item_b': 'object', 'bonus_cents': 'int64'}
Preview only (first 10 rows):
 table        item_a        item_b bonus_cents
bundle Rd1e83c40d290 R14062bafad03         271
bundle Rcabc56f3d592 R45c372fcda76         420
bundle R53ffd925fbfa R77b7948b94e4         214
bundle R12315a4dcd90 R5415495b66bf         210
bundle Rd16887582167 R99c4a58ed0e9         132
bundle R5415495b66bf Rcabc56f3d592         210
bundle R45c372fcda76 R57ed61d46b1b         166
bundle R53ffd925fbfa R12315a4dcd90         361
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 7}, "item_b": {"missing": 0, "unique_nonempty": 8}, "bonus_cents": {"missing": 0, "unique_nonempty": 7, "numeric_range": [132.0, 420.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "bonus_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "bundle", "matching_columns": [{"column": "table", "exact": 8, "prefix": 8, "contains": 8, "examples": ["bundle"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_02/export_02.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['table', 'resource', 'entry', 'amount', 'unit']
Parsed column types: {'table': 'object', 'resource': 'object', 'entry': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
          table resource       entry amount unit
capacity_ledger   AREA_B     opening 212000   ml
capacity_ledger   AREA_A     opening 233000   ml
capacity_ledger   AREA_C     opening 211000   ml
capacity_ledger   AREA_B reservation  -8000   ml
capacity_ledger   AREA_C reservation -11000   ml
capacity_ledger   AREA_A reservation -12000   ml
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "resource": {"missing": 0, "unique_nonempty": 3}, "entry": {"missing": 0, "unique_nonempty": 2}, "amount": {"missing": 0, "unique_nonempty": 6, "numeric_range": [-12000.0, 233000.0]}, "unit": {"missing": 0, "unique_nonempty": 1}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity_ledger", "matching_columns": [{"column": "table", "exact": 6, "prefix": 6, "contains": 6, "examples": ["capacity_ledger"]}], "exact_matching_columns": 1}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_03/export_03.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['table', 'category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'category': 'object', 'minimum_quantity': 'int64', 'maximum_quantity': 'int64', 'activation_fee_cents': 'int64'}
Preview only (first 10 rows):
   table category minimum_quantity maximum_quantity activation_fee_cents
category       G2                6               18                  136
category       G0                5               17                  306
category       G1                6               18                  257
category       G3                5               17                  358
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "category": {"missing": 0, "unique_nonempty": 4}, "minimum_quantity": {"missing": 0, "unique_nonempty": 2, "numeric_range": [5.0, 6.0]}, "maximum_quantity": {"missing": 0, "unique_nonempty": 2, "numeric_range": [17.0, 18.0]}, "activation_fee_cents": {"missing": 0, "unique_nonempty": 4, "numeric_range": [136.0, 358.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "activation_fee_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "category", "matching_columns": [{"column": "table", "exact": 4, "prefix": 4, "contains": 4, "examples": ["category"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_04/export_04.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 28
Columns: ['table', 'ref', 'kind', 'entity_id', 'display_name']
Parsed column types: {'table': 'object', 'ref': 'object', 'kind': 'object', 'entity_id': 'object', 'display_name': 'object'}
Preview only (first 10 rows):
   table           ref kind      entity_id       display_name
identity R14062bafad03 item RA14_OPTION_27      Geothermal AC
identity Rcca417551e0d item RA14_OPTION_11        Window Unit
identity Rf5c1f762749b item RA14_OPTION_17      Geothermal AC
identity R9fa17b3624a3 item RA14_OPTION_05         Central AC
identity Rdc2feb49418c item RA14_OPTION_01        Window Unit
identity R953de5726b93 item RA14_OPTION_07      Geothermal AC
identity Rcfec8c35a828 item RA14_OPTION_03       Split System
identity Rd1e83c40d290 item RA14_OPTION_15         Central AC
identity R5415495b66bf item RA14_OPTION_25         Central AC
identity R99c4a58ed0e9 item RA14_OPTION_09 Evaporative Cooler
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "ref": {"missing": 0, "unique_nonempty": 28}, "kind": {"missing": 0, "unique_nonempty": 1}, "entity_id": {"missing": 0, "unique_nonempty": 28}, "display_name": {"missing": 0, "unique_nonempty": 10}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item", "matching_columns": [{"column": "kind", "exact": 28, "prefix": 28, "contains": 28, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [{"column": "display_name", "exact": 0, "prefix": 0, "contains": 3, "examples": ["Portable Unit"]}], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_05/export_05.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 11
Columns: ['table', 'item_a', 'item_b']
Parsed column types: {'table': 'object', 'item_a': 'object', 'item_b': 'object'}
Preview only (first 10 rows):
       table        item_a        item_b
incompatible R77b7948b94e4 R12315a4dcd90
incompatible R77b7948b94e4 Re3ce1b0e9e02
incompatible Re3ce1b0e9e02 R57ed61d46b1b
incompatible R547a94bb1e2c R2caf5f536f7b
incompatible R77b7948b94e4 R14062bafad03
incompatible R99c4a58ed0e9 R77b7948b94e4
incompatible Rcabc56f3d592 R953de5726b93
incompatible R12315a4dcd90 R57ed61d46b1b
incompatible Rd1e83c40d290 R547a94bb1e2c
incompatible R77b7948b94e4 R2caf5f536f7b
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 7}, "item_b": {"missing": 0, "unique_nonempty": 8}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "incompatible", "matching_columns": [{"column": "table", "exact": 11, "prefix": 11, "contains": 11, "examples": ["incompatible"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_06/export_06.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 14
Columns: ['table', 'item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'configuration_id', 'location_id', 'unit_benefit_cents', 'item_fee_cents']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'category': 'object', 'authorized': 'int64', 'minimum_lot': 'int64', 'maximum_order': 'int64', 'configuration_id': 'object', 'location_id': 'object', 'unit_benefit_cents': 'int64', 'item_fee_cents': 'int64'}
Preview only (first 10 rows):
table      item_ref category authorized minimum_lot maximum_order configuration_id location_id unit_benefit_cents item_fee_cents
 item Rd16887582167       G2          1           2             7           CFG_03      AREA_B               1161            442
 item Rf5c1f762749b       G0          1           2             7           CFG_02      AREA_B                979            228
 item Rdc2feb49418c       G0          1           2             8           CFG_01      AREA_A               1548            149
 item R2caf5f536f7b       G2          1           2            12           CFG_02      AREA_A               1464            422
 item R99c4a58ed0e9       G0          1           2             8           CFG_01      AREA_C               1410            307
 item Rcfec8c35a828       G2          1           2             7           CFG_01      AREA_C               1287            304
 item Rcca417551e0d       G2          1           2            12           CFG_02      AREA_B               1530            104
 item R7a87faff4029       G3          1           2            11           CFG_01      AREA_B                944            252
 item R050d45e3aa87       G1          1           2            10           CFG_03      AREA_B                762            190
 item R45c372fcda76       G1          1           2             8           CFG_01      AREA_B               1034            201
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 14}, "category": {"missing": 0, "unique_nonempty": 4}, "authorized": {"missing": 0, "unique_nonempty": 1, "numeric_range": [1.0, 1.0]}, "minimum_lot": {"missing": 0, "unique_nonempty": 1, "numeric_range": [2.0, 2.0]}, "maximum_order": {"missing": 0, "unique_nonempty": 6, "numeric_range": [7.0, 12.0]}, "configuration_id": {"missing": 0, "unique_nonempty": 3}, "location_id": {"missing": 0, "unique_nonempty": 3}, "unit_benefit_cents": {"missing": 0, "unique_nonempty": 14, "numeric_range": [259.0, 1548.0]}, "item_fee_cents": {"missing": 0, "unique_nonempty": 14, "numeric_range": [104.0, 464.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "category", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item", "matching_columns": [{"column": "table", "exact": 14, "prefix": 14, "contains": 14, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "item_fee_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "location_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "maximum_order", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit_benefit_cents", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_01/export_07.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 14
Columns: ['table', 'item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'configuration_id', 'location_id', 'unit_benefit_cents', 'item_fee_cents']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'category': 'object', 'authorized': 'int64', 'minimum_lot': 'int64', 'maximum_order': 'int64', 'configuration_id': 'object', 'location_id': 'object', 'unit_benefit_cents': 'int64', 'item_fee_cents': 'int64'}
Preview only (first 10 rows):
table      item_ref category authorized minimum_lot maximum_order configuration_id location_id unit_benefit_cents item_fee_cents
 item R14062bafad03       G2          0           2            12           CFG_03      AREA_C               9000            334
 item Rd1e83c40d290       G2          1           2            12           CFG_02      AREA_C                730            170
 item R953de5726b93       G2          1           2             8           CFG_01      AREA_A                755            146
 item R57ed61d46b1b       G0          0           2            13           CFG_03      AREA_C               9000            219
 item R9fa17b3624a3       G0          1           2            12           CFG_01      AREA_B                911            213
 item R77b7948b94e4       G0          1           2             9           CFG_02      AREA_A                568            407
 item R5415495b66bf       G0          0           2            10           CFG_03      AREA_A               9000            183
 item Rc1af75b326dc       G3          0           2             8           CFG_02      AREA_B               9000            266
 item Rf1ec5c99c962       G3          1           2            12           CFG_02      AREA_A                573            163
 item R12315a4dcd90       G1          1           2            13           CFG_02      AREA_C                844            384
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 14}, "category": {"missing": 0, "unique_nonempty": 4}, "authorized": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}, "minimum_lot": {"missing": 0, "unique_nonempty": 1, "numeric_range": [2.0, 2.0]}, "maximum_order": {"missing": 0, "unique_nonempty": 5, "numeric_range": [8.0, 13.0]}, "configuration_id": {"missing": 0, "unique_nonempty": 3}, "location_id": {"missing": 0, "unique_nonempty": 3}, "unit_benefit_cents": {"missing": 0, "unique_nonempty": 11, "numeric_range": [568.0, 9000.0]}, "item_fee_cents": {"missing": 0, "unique_nonempty": 14, "numeric_range": [146.0, 497.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "category", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item", "matching_columns": [{"column": "table", "exact": 14, "prefix": 14, "contains": 14, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "item_fee_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "location_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "maximum_order", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit_benefit_cents", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_02/export_08.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 28
Columns: ['table', 'item_ref', 'historical_sales', 'forecast_units', 'margin_percent']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'historical_sales': 'int64', 'forecast_units': 'int64', 'margin_percent': 'int64'}
Preview only (first 10 rows):
 table      item_ref historical_sales forecast_units margin_percent
market R2caf5f536f7b                1              1             22
market R953de5726b93                2              1             34
market R77b7948b94e4                3              3             12
market R9fa17b3624a3                3              3             27
market R57ed61d46b1b                1              2             28
market R99c4a58ed0e9                1              3             20
market Rd16887582167                2              3             12
market Rdc2feb49418c                1              3             31
market Rd1e83c40d290                2              1             27
market Rf5c1f762749b                1              2             27
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 28}, "historical_sales": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "forecast_units": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "margin_percent": {"missing": 0, "unique_nonempty": 19, "numeric_range": [5.0, 34.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_03/export_09.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 16
Columns: ['table', 'item_ref', 'prerequisite_ref']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'prerequisite_ref': 'object'}
Preview only (first 10 rows):
   table      item_ref prerequisite_ref
requires Rf5c1f762749b    R53ffd925fbfa
requires R77b7948b94e4    Re3ce1b0e9e02
requires Rd16887582167    R547a94bb1e2c
requires R2caf5f536f7b    R953de5726b93
requires R14062bafad03    Rdc2feb49418c
requires Rd1e83c40d290    R9fa17b3624a3
requires R57ed61d46b1b    Re3ce1b0e9e02
requires R5415495b66bf    R99c4a58ed0e9
requires R12315a4dcd90    Rdc2feb49418c
requires R050d45e3aa87    R53ffd925fbfa
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 16}, "prerequisite_ref": {"missing": 0, "unique_nonempty": 9}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "prerequisite_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "requires", "matching_columns": [{"column": "table", "exact": 16, "prefix": 16, "contains": 16, "examples": ["requires"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_04/export_10.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 14
Columns: ['table', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
table      item_ref resource amount unit
usage Rdc2feb49418c   AREA_A   8000   ml
usage Rf1ec5c99c962   AREA_A   3000   ml
usage R953de5726b93   AREA_A   8000   ml
usage R45aa14bd7960   AREA_A   4000   ml
usage R93243a2d30f2   AREA_A   7000   ml
usage R7a87faff4029   AREA_B   9000   ml
usage Rd16887582167   AREA_B   4000   ml
usage Rb96d2724c33e   AREA_B   5000   ml
usage R050d45e3aa87   AREA_B   8000   ml
usage R9fa17b3624a3   AREA_B   3000   ml
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 14}, "resource": {"missing": 0, "unique_nonempty": 3}, "amount": {"missing": 0, "unique_nonempty": 7, "numeric_range": [3000.0, 9000.0]}, "unit": {"missing": 0, "unique_nonempty": 1}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 14, "prefix": 14, "contains": 14, "examples": ["usage"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_05/export_11.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 14
Columns: ['table', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
table      item_ref resource amount unit
usage R77b7948b94e4   AREA_A   7000   ml
usage Re3ce1b0e9e02   AREA_A   8000   ml
usage Rd2a478d62ea8   AREA_A   8000   ml
usage R5415495b66bf   AREA_A   3000   ml
usage R2caf5f536f7b   AREA_A   4000   ml
usage Rc1af75b326dc   AREA_B   8000   ml
usage R45c372fcda76   AREA_B   9000   ml
usage Rf5c1f762749b   AREA_B   3000   ml
usage Rcca417551e0d   AREA_B   4000   ml
usage Rcfec8c35a828   AREA_C   9000   ml
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 14}, "resource": {"missing": 0, "unique_nonempty": 3}, "amount": {"missing": 0, "unique_nonempty": 7, "numeric_range": [3000.0, 9000.0]}, "unit": {"missing": 0, "unique_nonempty": 1}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 14, "prefix": 14, "contains": 14, "examples": ["usage"]}], "exact_matching_columns": 1}]