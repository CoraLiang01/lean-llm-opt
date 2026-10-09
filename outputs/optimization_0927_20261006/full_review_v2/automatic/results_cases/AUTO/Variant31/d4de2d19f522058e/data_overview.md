File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_01.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['table', 'item_a', 'item_b', 'bonus_cents']
Parsed column types: {'table': 'object', 'item_a': 'object', 'item_b': 'object', 'bonus_cents': 'int64'}
Preview only (first 10 rows):
 table        item_a        item_b bonus_cents
bundle Rf79dba335287 R8870b8b0af60         141
bundle Rfa17d92465fe Rac94dfa786d2         183
bundle Rac94dfa786d2 Rc687626686b9         356
bundle R4c746990c118 R0e4773ef6644         504
bundle Rf3fcb229dd19 R07756e8a5f84         274
bundle R6223330542a2 Rd92555c3918d         501
bundle Rf5a912d3e6b6 R965543941518         249
bundle R3bbdcaedffa4 Rf3fcb229dd19         142
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 8}, "item_b": {"missing": 0, "unique_nonempty": 8}, "bonus_cents": {"missing": 0, "unique_nonempty": 8, "numeric_range": [141.0, 504.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "bundle", "matching_columns": [{"column": "table", "exact": 8, "prefix": 8, "contains": 8, "examples": ["bundle"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_02.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['table', 'resource', 'entry', 'amount', 'unit']
Parsed column types: {'table': 'object', 'resource': 'object', 'entry': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
          table resource       entry amount   unit
capacity_ledger    power     opening 252000     wh
capacity_ledger    labor     opening  16020 minute
capacity_ledger    space     opening 367000     ml
capacity_ledger    space reservation -10000     ml
capacity_ledger    power reservation  -7000     wh
capacity_ledger    labor reservation   -540 minute
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "resource": {"missing": 0, "unique_nonempty": 3}, "entry": {"missing": 0, "unique_nonempty": 2}, "amount": {"missing": 0, "unique_nonempty": 6, "numeric_range": [-10000.0, 367000.0]}, "unit": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity_ledger", "matching_columns": [{"column": "table", "exact": 6, "prefix": 6, "contains": 6, "examples": ["capacity_ledger"]}], "exact_matching_columns": 1}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_03.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['table', 'category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'category': 'object', 'minimum_quantity': 'int64', 'maximum_quantity': 'int64', 'activation_fee_cents': 'int64'}
Preview only (first 10 rows):
   table category minimum_quantity maximum_quantity activation_fee_cents
category       G3               11               23                  386
category       G1                5               17                  152
category       G2                9               21                  170
category       G0                9               21                  201
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "category": {"missing": 0, "unique_nonempty": 4}, "minimum_quantity": {"missing": 0, "unique_nonempty": 3, "numeric_range": [5.0, 11.0]}, "maximum_quantity": {"missing": 0, "unique_nonempty": 3, "numeric_range": [17.0, 23.0]}, "activation_fee_cents": {"missing": 0, "unique_nonempty": 4, "numeric_range": [152.0, 386.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "category", "matching_columns": [{"column": "table", "exact": 4, "prefix": 4, "contains": 4, "examples": ["category"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_04/export_04.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 25
Columns: ['table', 'ref', 'kind', 'entity_id', 'display_name']
Parsed column types: {'table': 'object', 'ref': 'object', 'kind': 'object', 'entity_id': 'object', 'display_name': 'object'}
Preview only (first 10 rows):
   table           ref kind     entity_id  display_name
identity R699b0becd768 item RA3_OPTION_02           SUV
identity Rf3fcb229dd19 item RA3_OPTION_22    Camper Van
identity R0e4773ef6644 item RA3_OPTION_18  Pickup Truck
identity R460b61388994 item RA3_OPTION_10    Hybrid Car
identity Rbb201ffffa62 item RA3_OPTION_08 Station Wagon
identity R965543941518 item RA3_OPTION_24    Motorcycle
identity R43da61540750 item RA3_OPTION_04   Convertible
identity Rfa17d92465fe item RA3_OPTION_20    Muscle Car
identity R07756e8a5f84 item RA3_OPTION_12    Sports Car
identity Rf79dba335287 item RA3_OPTION_06         Coupe
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "ref": {"missing": 0, "unique_nonempty": 25}, "kind": {"missing": 0, "unique_nonempty": 1}, "entity_id": {"missing": 0, "unique_nonempty": 25}, "display_name": {"missing": 0, "unique_nonempty": 25}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item", "matching_columns": [{"column": "kind", "exact": 25, "prefix": 25, "contains": 25, "examples": ["item"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_05/export_05.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 10
Columns: ['table', 'item_a', 'item_b']
Parsed column types: {'table': 'object', 'item_a': 'object', 'item_b': 'object'}
Preview only (first 10 rows):
       table        item_a        item_b
incompatible Rf1044fcaeb7a R3bbdcaedffa4
incompatible R43da61540750 Rac94dfa786d2
incompatible Rf1044fcaeb7a Rfa17d92465fe
incompatible R20246322f2b5 Rbb201ffffa62
incompatible Rb589859aac9b Rd92555c3918d
incompatible R07756e8a5f84 R4c746990c118
incompatible Rae2b555cc8ad Rf1044fcaeb7a
incompatible R965543941518 R8870b8b0af60
incompatible R8870b8b0af60 Rf5a912d3e6b6
incompatible Rf72785750151 Rae2b555cc8ad
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 9}, "item_b": {"missing": 0, "unique_nonempty": 10}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "incompatible", "matching_columns": [{"column": "table", "exact": 10, "prefix": 10, "contains": 10, "examples": ["incompatible"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_06/export_06.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 25
Columns: ['table', 'item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'configuration_id', 'unit_benefit_cents', 'item_fee_cents']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'category': 'object', 'authorized': 'int64', 'minimum_lot': 'int64', 'maximum_order': 'int64', 'configuration_id': 'object', 'unit_benefit_cents': 'int64', 'item_fee_cents': 'int64'}
Preview only (first 10 rows):
table      item_ref category authorized minimum_lot maximum_order configuration_id unit_benefit_cents item_fee_cents
 item R699b0becd768       G1          1           2             7           CFG_01               1196            277
 item R965543941518       G3          0           2             8           CFG_01               9000            174
 item R460b61388994       G1          1           2            10           CFG_01               1358            380
 item Rf79dba335287       G1          1           2            10           CFG_01                346            277
 item Rb589859aac9b       G3          1           2            10           CFG_01                404            430
 item R0e4773ef6644       G1          1           2            13           CFG_01               1593            402
 item Rbb201ffffa62       G3          1           2            10           CFG_01               1243            295
 item Rfa17d92465fe       G3          1           2            11           CFG_01               1059            273
 item R07756e8a5f84       G3          1           2             7           CFG_01               1440            161
 item Rf3fcb229dd19       G1          1           2             8           CFG_01                460            136
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 25}, "category": {"missing": 0, "unique_nonempty": 4}, "authorized": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}, "minimum_lot": {"missing": 0, "unique_nonempty": 1, "numeric_range": [2.0, 2.0]}, "maximum_order": {"missing": 0, "unique_nonempty": 6, "numeric_range": [7.0, 13.0]}, "configuration_id": {"missing": 0, "unique_nonempty": 1}, "unit_benefit_cents": {"missing": 0, "unique_nonempty": 22, "numeric_range": [319.0, 9000.0]}, "item_fee_cents": {"missing": 0, "unique_nonempty": 24, "numeric_range": [119.0, 481.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "authorized", "matching_columns": [], "exact_matching_columns": 0}, {"term": "category", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item", "matching_columns": [{"column": "table", "exact": 25, "prefix": 25, "contains": 25, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "item_fee_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "maximum_order", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit_benefit_cents", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_07.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 25
Columns: ['table', 'item_ref', 'historical_sales', 'forecast_units', 'margin_percent']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'historical_sales': 'int64', 'forecast_units': 'int64', 'margin_percent': 'int64'}
Preview only (first 10 rows):
 table      item_ref historical_sales forecast_units margin_percent
market R460b61388994                1              2             24
market R699b0becd768                1              2             13
market Rf3fcb229dd19                1              1             25
market R07756e8a5f84                1              3             29
market Rbb201ffffa62                3              1              5
market R43da61540750                2              1             18
market Rac94dfa786d2                2              1             21
market R965543941518                1              2             22
market Rfa17d92465fe                1              1              7
market Rf79dba335287                2              1             16
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 25}, "historical_sales": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "forecast_units": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "margin_percent": {"missing": 0, "unique_nonempty": 20, "numeric_range": [5.0, 29.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: []

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_08.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 13
Columns: ['table', 'item_ref', 'prerequisite_ref']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'prerequisite_ref': 'object'}
Preview only (first 10 rows):
   table      item_ref prerequisite_ref
requires R20246322f2b5    R3bbdcaedffa4
requires Rf72785750151    R699b0becd768
requires Rc687626686b9    R3bbdcaedffa4
requires R4c746990c118    R699b0becd768
requires Raf59552d6773    Rae2b555cc8ad
requires Rf1044fcaeb7a    Rbb201ffffa62
requires Rf5a912d3e6b6    Rbb201ffffa62
requires Rac94dfa786d2    R699b0becd768
requires Rfa17d92465fe    R8870b8b0af60
requires R0e4773ef6644    R07756e8a5f84
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 13}, "prerequisite_ref": {"missing": 0, "unique_nonempty": 8}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "requires", "matching_columns": [{"column": "table", "exact": 13, "prefix": 13, "contains": 13, "examples": ["requires"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_09.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 75
Columns: ['table', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
table      item_ref resource amount unit
usage R0e4773ef6644    power   3000   wh
usage Rf79dba335287    power   8000   wh
usage R6223330542a2    power   3000   wh
usage R8870b8b0af60    power   4000   wh
usage Rf1044fcaeb7a    power   6000   wh
usage Rf3fcb229dd19    power   5000   wh
usage Rf72785750151    power   5000   wh
usage R43da61540750    power   5000   wh
usage R965543941518    power   7000   wh
usage Rd92555c3918d    power   3000   wh
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 25}, "resource": {"missing": 0, "unique_nonempty": 3}, "amount": {"missing": 0, "unique_nonempty": 16, "numeric_range": [120.0, 9000.0]}, "unit": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 75, "prefix": 75, "contains": 75, "examples": ["usage"]}], "exact_matching_columns": 1}]