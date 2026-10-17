File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_01.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 60
Columns: ['table', 'item_ref', 'component', 'amount', 'currency']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'component': 'object', 'amount': 'int64', 'currency': 'object'}
Preview only (first 10 rows):
  table      item_ref     component amount currency
benefit R5adef152c6ae  item7_rebate    -29      USD
benefit Ra4886770fce2   item24_base   1133      USD
benefit Rb968fc7263d0 item19_rebate     -7      USD
benefit R49f121c38707   item18_base  57400      JPY
benefit Racfc6fa104e0 item22_rebate  -1600      JPY
benefit R366d30e9231a item25_rebate    -27      USD
benefit R2c34c7e3f259   item12_base  91800      JPY
benefit Rb105f8f13329 item10_rebate    -20      EUR
benefit R75a6d1f194cd   item15_base   3090      EUR
benefit Ra37606a6bb32   item21_base  18094      EUR
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 30}, "component": {"missing": 0, "unique_nonempty": 60}, "amount": {"missing": 0, "unique_nonempty": 55, "numeric_range": [-3700.0, 902700.0]}, "currency": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "benefit", "matching_columns": [{"column": "table", "exact": 60, "prefix": 60, "contains": 60, "examples": ["benefit"]}], "exact_matching_columns": 1}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usd", "matching_columns": [{"column": "currency", "exact": 18, "prefix": 18, "contains": 18, "examples": ["USD"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_02.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['table', 'item_a', 'item_b', 'bonus_cents']
Parsed column types: {'table': 'object', 'item_a': 'object', 'item_b': 'object', 'bonus_cents': 'int64'}
Preview only (first 10 rows):
 table        item_a        item_b bonus_cents
bundle R5f3200ccfff2 R6859cf92a978         303
bundle R10f7da03ae36 R75a6d1f194cd         208
bundle Ra37606a6bb32 R94e8dab78c7f         409
bundle R9a15b393da03 R6660529b82e0         581
bundle Rfe6ec2e40743 Racfc6fa104e0         138
bundle Rf75236ab99b1 R342034fb919b         329
bundle R94e8dab78c7f R75a6d1f194cd         563
bundle R366d30e9231a R2cdd7ae5edde         600
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 8}, "item_b": {"missing": 0, "unique_nonempty": 7}, "bonus_cents": {"missing": 0, "unique_nonempty": 8, "numeric_range": [138.0, 600.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "bonus_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "bundle", "matching_columns": [{"column": "table", "exact": 8, "prefix": 8, "contains": 8, "examples": ["bundle"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_03.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 6
Columns: ['table', 'resource', 'entry', 'amount', 'unit']
Parsed column types: {'table': 'object', 'resource': 'object', 'entry': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
          table resource       entry amount unit
capacity_ledger  CONSOLE     opening 216000   MB
capacity_ledger       PC     opening 276000   MB
capacity_ledger   MOBILE     opening 268000   MB
capacity_ledger  CONSOLE reservation  -6000   MB
capacity_ledger   MOBILE reservation -10000   MB
capacity_ledger       PC reservation  -8000   MB
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "resource": {"missing": 0, "unique_nonempty": 3}, "entry": {"missing": 0, "unique_nonempty": 2}, "amount": {"missing": 0, "unique_nonempty": 6, "numeric_range": [-10000.0, 276000.0]}, "unit": {"missing": 0, "unique_nonempty": 1}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "capacity_ledger", "matching_columns": [{"column": "table", "exact": 6, "prefix": 6, "contains": 6, "examples": ["capacity_ledger"]}], "exact_matching_columns": 1}, {"term": "console", "matching_columns": [{"column": "resource", "exact": 2, "prefix": 2, "contains": 2, "examples": ["CONSOLE"]}], "exact_matching_columns": 1}, {"term": "mb", "matching_columns": [{"column": "unit", "exact": 6, "prefix": 6, "contains": 6, "examples": ["MB"]}], "exact_matching_columns": 1}, {"term": "mobile", "matching_columns": [{"column": "resource", "exact": 2, "prefix": 2, "contains": 2, "examples": ["MOBILE"]}], "exact_matching_columns": 1}, {"term": "pc", "matching_columns": [{"column": "resource", "exact": 2, "prefix": 2, "contains": 2, "examples": ["PC"]}], "exact_matching_columns": 1}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_04.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 4
Columns: ['table', 'category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'category': 'object', 'minimum_quantity': 'int64', 'maximum_quantity': 'int64', 'activation_fee_cents': 'int64'}
Preview only (first 10 rows):
   table category minimum_quantity maximum_quantity activation_fee_cents
category       G0                6               18                  215
category       G2                8               20                  223
category       G1                7               19                  193
category       G3               12               24                  185
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "category": {"missing": 0, "unique_nonempty": 4}, "minimum_quantity": {"missing": 0, "unique_nonempty": 4, "numeric_range": [6.0, 12.0]}, "maximum_quantity": {"missing": 0, "unique_nonempty": 4, "numeric_range": [18.0, 24.0]}, "activation_fee_cents": {"missing": 0, "unique_nonempty": 4, "numeric_range": [185.0, 223.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "activation_fee_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "category", "matching_columns": [{"column": "table", "exact": 4, "prefix": 4, "contains": 4, "examples": ["category"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_05.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 3
Columns: ['table', 'currency', 'usd_cents_numerator', 'denominator']
Parsed column types: {'table': 'object', 'currency': 'object', 'usd_cents_numerator': 'int64', 'denominator': 'int64'}
Preview only (first 10 rows):
table currency usd_cents_numerator denominator
   fx      JPY                   1         100
   fx      EUR                   1           2
   fx      USD                   1           1
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "currency": {"missing": 0, "unique_nonempty": 3}, "usd_cents_numerator": {"missing": 0, "unique_nonempty": 1, "numeric_range": [1.0, 1.0]}, "denominator": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 100.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "denominator", "matching_columns": [], "exact_matching_columns": 0}, {"term": "fx", "matching_columns": [{"column": "table", "exact": 3, "prefix": 3, "contains": 3, "examples": ["fx"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usd", "matching_columns": [{"column": "currency", "exact": 1, "prefix": 1, "contains": 1, "examples": ["USD"]}], "exact_matching_columns": 1}, {"term": "usd_cents_numerator", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_06.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 30
Columns: ['table', 'ref', 'kind', 'entity_id', 'display_name']
Parsed column types: {'table': 'object', 'ref': 'object', 'kind': 'object', 'entity_id': 'object', 'display_name': 'object'}
Preview only (first 10 rows):
   table           ref kind      entity_id display_name
identity R59cecdfab692 item RA13_OPTION_21      Shooter
identity Ra4886770fce2 item RA13_OPTION_25     Fighting
identity R94e8dab78c7f item RA13_OPTION_01       Racing
identity Racfc6fa104e0 item RA13_OPTION_23   Simulation
identity R9a15b393da03 item RA13_OPTION_27     Survival
identity R49ac29054b96 item RA13_OPTION_29      Sandbox
identity R49f121c38707 item RA13_OPTION_19    Adventure
identity R2cdd7ae5edde item RA13_OPTION_03       Action
identity R2c34c7e3f259 item RA13_OPTION_13       Horror
identity Rc19649816c5e item RA13_OPTION_09       Puzzle
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "ref": {"missing": 0, "unique_nonempty": 30}, "kind": {"missing": 0, "unique_nonempty": 1}, "entity_id": {"missing": 0, "unique_nonempty": 30}, "display_name": {"missing": 0, "unique_nonempty": 15}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item", "matching_columns": [{"column": "kind", "exact": 30, "prefix": 30, "contains": 30, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_07.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 11
Columns: ['table', 'item_a', 'item_b']
Parsed column types: {'table': 'object', 'item_a': 'object', 'item_b': 'object'}
Preview only (first 10 rows):
       table        item_a        item_b
incompatible R6859cf92a978 R6660529b82e0
incompatible R6660529b82e0 Ra37606a6bb32
incompatible Rf75236ab99b1 Ra37606a6bb32
incompatible R49ac29054b96 Racfc6fa104e0
incompatible Racfc6fa104e0 R8bf0d675a614
incompatible R4702703af2e4 R8facc1d99668
incompatible R49ac29054b96 R8bf0d675a614
incompatible Racfc6fa104e0 R94e8dab78c7f
incompatible Rf75236ab99b1 R6859cf92a978
incompatible R4bae4d328f7e R9a15b393da03
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 8}, "item_b": {"missing": 0, "unique_nonempty": 9}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "incompatible", "matching_columns": [{"column": "table", "exact": 11, "prefix": 11, "contains": 11, "examples": ["incompatible"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_08.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 30
Columns: ['table', 'item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'configuration_id', 'location_id']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'category': 'object', 'authorized': 'int64', 'minimum_lot': 'int64', 'maximum_order': 'int64', 'configuration_id': 'object', 'location_id': 'object'}
Preview only (first 10 rows):
table      item_ref category authorized minimum_lot maximum_order configuration_id location_id
 item R4bae4d328f7e       G2          1           2            11           CFG_01          PC
 item R49ac29054b96       G0          1           2             9           CFG_02     CONSOLE
 item R49f121c38707       G2          1           2            12           CFG_02          PC
 item Rc19649816c5e       G0          1           2            10           CFG_01      MOBILE
 item R10f7da03ae36       G0          1           2             9           CFG_01     CONSOLE
 item R84e0f5c1d754       G2          1           2            13           CFG_01      MOBILE
 item R94e8dab78c7f       G0          1           2            12           CFG_01          PC
 item R59cecdfab692       G0          1           2             8           CFG_02      MOBILE
 item Ra4886770fce2       G0          1           2            10           CFG_02          PC
 item R9a15b393da03       G2          1           2             9           CFG_02      MOBILE
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 30}, "category": {"missing": 0, "unique_nonempty": 4}, "authorized": {"missing": 0, "unique_nonempty": 2, "numeric_range": [0.0, 1.0]}, "minimum_lot": {"missing": 0, "unique_nonempty": 1, "numeric_range": [2.0, 2.0]}, "maximum_order": {"missing": 0, "unique_nonempty": 7, "numeric_range": [7.0, 13.0]}, "configuration_id": {"missing": 0, "unique_nonempty": 2}, "location_id": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "category", "matching_columns": [], "exact_matching_columns": 0}, {"term": "console", "matching_columns": [{"column": "location_id", "exact": 10, "prefix": 10, "contains": 10, "examples": ["CONSOLE"]}], "exact_matching_columns": 1}, {"term": "item", "matching_columns": [{"column": "table", "exact": 30, "prefix": 30, "contains": 30, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "location_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "maximum_order", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "mobile", "matching_columns": [{"column": "location_id", "exact": 10, "prefix": 10, "contains": 10, "examples": ["MOBILE"]}], "exact_matching_columns": 1}, {"term": "pc", "matching_columns": [{"column": "location_id", "exact": 10, "prefix": 10, "contains": 10, "examples": ["PC"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_09.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 30
Columns: ['table', 'item_ref', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'activation_fee_cents': 'int64'}
Preview only (first 10 rows):
   table      item_ref activation_fee_cents
item_fee Racfc6fa104e0                  443
item_fee R4bae4d328f7e                  226
item_fee R49f121c38707                  325
item_fee R9a15b393da03                  416
item_fee Ra4886770fce2                  159
item_fee R84e0f5c1d754                  197
item_fee R59cecdfab692                  232
item_fee Rb105f8f13329                  415
item_fee R94e8dab78c7f                  146
item_fee Rc19649816c5e                  174
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 30}, "activation_fee_cents": {"missing": 0, "unique_nonempty": 28, "numeric_range": [108.0, 475.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "activation_fee_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item_fee", "matching_columns": [{"column": "table", "exact": 30, "prefix": 30, "contains": 30, "examples": ["item_fee"]}], "exact_matching_columns": 1}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_10.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 30
Columns: ['table', 'item_ref', 'historical_sales', 'forecast_units', 'margin_percent']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'historical_sales': 'int64', 'forecast_units': 'int64', 'margin_percent': 'int64'}
Preview only (first 10 rows):
 table      item_ref historical_sales forecast_units margin_percent
market R59cecdfab692                1              1             19
market Ra4886770fce2                3              1             31
market R62b480a38aa1                3              2             23
market R2c34c7e3f259                2              1              7
market R94e8dab78c7f                3              3             14
market R49ac29054b96                3              3             29
market Rb105f8f13329                2              2             23
market R9a15b393da03                1              3             29
market Racfc6fa104e0                2              2             33
market R4bae4d328f7e                3              1             28
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 30}, "historical_sales": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "forecast_units": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "margin_percent": {"missing": 0, "unique_nonempty": 21, "numeric_range": [5.0, 35.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_11.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 18
Columns: ['table', 'item_ref', 'prerequisite_ref']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'prerequisite_ref': 'object'}
Preview only (first 10 rows):
   table      item_ref prerequisite_ref
requires R49f121c38707    R4bae4d328f7e
requires Ra4886770fce2    Rb105f8f13329
requires R62b480a38aa1    R5f3200ccfff2
requires R59cecdfab692    R8facc1d99668
requires R2c34c7e3f259    Rb105f8f13329
requires R84e0f5c1d754    R10f7da03ae36
requires R9a15b393da03    R5adef152c6ae
requires Racfc6fa104e0    R2cdd7ae5edde
requires R49ac29054b96    Rb105f8f13329
requires R366d30e9231a    R5f3200ccfff2
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 18}, "prerequisite_ref": {"missing": 0, "unique_nonempty": 10}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "prerequisite_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "requires", "matching_columns": [{"column": "table", "exact": 18, "prefix": 18, "contains": 18, "examples": ["requires"]}], "exact_matching_columns": 1}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_12.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 30
Columns: ['table', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
table      item_ref resource amount unit
usage R4a5ff3f69427       PC      8   GB
usage R75a6d1f194cd       PC      8   GB
usage R49f121c38707       PC      3   GB
usage R4bae4d328f7e       PC      4   GB
usage R2c34c7e3f259       PC      7   GB
usage R94e8dab78c7f       PC      7   GB
usage Ra37606a6bb32       PC      9   GB
usage R8bf0d675a614       PC      9   GB
usage R8facc1d99668       PC      3   GB
usage Ra4886770fce2       PC      3   GB
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 30}, "resource": {"missing": 0, "unique_nonempty": 3}, "amount": {"missing": 0, "unique_nonempty": 8, "numeric_range": [2.0, 9.0]}, "unit": {"missing": 0, "unique_nonempty": 1}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount", "matching_columns": [], "exact_matching_columns": 0}, {"term": "console", "matching_columns": [{"column": "resource", "exact": 10, "prefix": 10, "contains": 10, "examples": ["CONSOLE"]}], "exact_matching_columns": 1}, {"term": "gb", "matching_columns": [{"column": "unit", "exact": 30, "prefix": 30, "contains": 30, "examples": ["GB"]}], "exact_matching_columns": 1}, {"term": "item_ref", "matching_columns": [], "exact_matching_columns": 0}, {"term": "mobile", "matching_columns": [{"column": "resource", "exact": 10, "prefix": 10, "contains": 10, "examples": ["MOBILE"]}], "exact_matching_columns": 1}, {"term": "pc", "matching_columns": [{"column": "resource", "exact": 10, "prefix": 10, "contains": 10, "examples": ["PC"]}], "exact_matching_columns": 1}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 30, "prefix": 30, "contains": 30, "examples": ["usage"]}], "exact_matching_columns": 1}]