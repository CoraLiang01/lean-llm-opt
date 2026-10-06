File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_01/export_01.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 26
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'prerequisite_ref']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'prerequisite_ref': 'object'}
Preview only (first 10 rows):
   table tenant     record_id revision effective_date action      item_ref prerequisite_ref
requires  NORTH requires:0001        2     2026-02-21 UPSERT R747b4b07f2df    R7955a6f70abc
requires  SOUTH requires:0008        2     2026-02-21 UPSERT Rc2e7f1ec75e0    R65d8546a7fb6
requires  NORTH requires:0013        1     2026-01-21 UPSERT R7308b473a5a8    R6007124a9f06
requires  NORTH requires:0017        2     2026-02-21 UPSERT R9197b70e3d09    R148f899e57f6
requires  NORTH requires:0001        1     2026-01-21 UPSERT R747b4b07f2df    R7955a6f70abc
requires  NORTH requires:0011        1     2026-01-21 UPSERT Ra5833cedea5c    R1820116348cd
requires  NORTH requires:0017        1     2026-01-21 UPSERT R9197b70e3d09    R148f899e57f6
requires  NORTH requires:0009        1     2026-01-21 UPSERT Rd05f0610c770    R65d8546a7fb6
requires  NORTH requires:0015        2     2026-02-21 UPSERT Rf9db851ed8bb    R2b2709069f4c
requires  NORTH requires:0009        2     2026-02-21 UPSERT Rd05f0610c770    R65d8546a7fb6
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 13}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 13}, "prerequisite_ref": {"missing": 0, "unique_nonempty": 9}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "north", "matching_columns": [{"column": "tenant", "exact": 23, "prefix": 23, "contains": 23, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "requires", "matching_columns": [{"column": "table", "exact": 26, "prefix": 26, "contains": 26, "examples": ["requires"]}, {"column": "record_id", "exact": 0, "prefix": 26, "contains": 26, "examples": ["requires:0001", "requires:0008", "requires:0013"]}], "exact_matching_columns": 1}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_02/export_02.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 41
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'ref', 'kind', 'entity_id']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'ref': 'object', 'kind': 'object', 'entity_id': 'object'}
Preview only (first 10 rows):
   table tenant     record_id revision effective_date action           ref kind entity_id
identity  NORTH identity:0001        2     2026-02-21 UPSERT Ref851ccaa158 item       I01
identity  NORTH identity:0003        1     2026-01-21 UPSERT R7955a6f70abc item       I03
identity  NORTH identity:0003        2     2026-02-21 UPSERT R7955a6f70abc item       I03
identity  NORTH identity:0009        2     2026-02-21 UPSERT R1b36613ee493 item       I09
identity  NORTH identity:0015        2     2026-02-21 UPSERT Re7d589eb4737 item       I15
identity  NORTH identity:0007        2     2026-02-21 UPSERT R1820116348cd item       I07
identity  SOUTH identity:0008        2     2026-02-21 UPSERT Rf6f2a500079e item       I08
identity  NORTH identity:0013        2     2026-02-21 UPSERT R747b4b07f2df item       I13
identity  NORTH identity:0017        1     2026-01-21 UPSERT R0f58f5015d0d item       I17
identity  NORTH identity:0023        1     2026-01-21 UPSERT Ra5833cedea5c item       I23
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 20}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "ref": {"missing": 0, "unique_nonempty": 20}, "kind": {"missing": 0, "unique_nonempty": 1}, "entity_id": {"missing": 0, "unique_nonempty": 20}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item", "matching_columns": [{"column": "kind", "exact": 41, "prefix": 41, "contains": 41, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 37, "prefix": 37, "contains": 37, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_03/export_03.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 45
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'activation_fee_cents': 'float64'}
Preview only (first 10 rows):
   table tenant     record_id revision effective_date action      item_ref activation_fee_cents
item_fee  NORTH item_fee:0018        2     2026-02-21 UPSERT Rfb39090aa5c5                  490
item_fee  NORTH item_fee:0012        2     2026-02-21 UPSERT R665d41e9e962                  499
item_fee  NORTH item_fee:0016        1     2026-01-21 UPSERT R3a11be59ddaa                  501
item_fee  NORTH item_fee:0006        2     2026-02-21 UPSERT R148f899e57f6                  492
item_fee  SOUTH item_fee:0020        2     2026-02-21 UPSERT Rc2e7f1ec75e0                  132
item_fee  NORTH item_fee:0008        1     2026-01-21 UPSERT Rf6f2a500079e                  224
item_fee  NORTH item_fee:0016        2     2026-02-21 UPSERT R3a11be59ddaa                  494
item_fee  NORTH item_fee:0008        2     2026-02-21 UPSERT Rf6f2a500079e                  217
item_fee  NORTH item_fee:0028        2     2026-02-21 UPSERT R7dcc05e5ca15                  336
item_fee  NORTH item_fee:0022        2     2026-02-21 UPSERT Rc07011e6ab2d                  195
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 17}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_ref": {"missing": 1, "unique_nonempty": 16}, "activation_fee_cents": {"missing": 1, "unique_nonempty": 35, "numeric_range": [123.0, 506.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "item_fee", "matching_columns": [{"column": "table", "exact": 45, "prefix": 45, "contains": 45, "examples": ["item_fee"]}, {"column": "record_id", "exact": 0, "prefix": 45, "contains": 45, "examples": ["item_fee:0018", "item_fee:0012", "item_fee:0016"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 41, "prefix": 41, "contains": 41, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_04/export_04.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 11
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_a', 'item_b', 'bonus_cents']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_a': 'object', 'item_b': 'object', 'bonus_cents': 'int64'}
Preview only (first 10 rows):
 table tenant   record_id revision effective_date action        item_a        item_b bonus_cents
bundle  NORTH bundle:0001        1     2026-01-21 UPSERT R7308b473a5a8 Rf6f2a500079e         136
bundle  SOUTH bundle:0000        2     2026-02-21 UPSERT R148f899e57f6 R698808ad9a96         174
bundle  NORTH bundle:0007        1     2026-01-21 UPSERT R148f899e57f6 Rc07011e6ab2d         276
bundle  NORTH bundle:0005        3     2026-03-23 UPSERT R698808ad9a96 R07e87ed82812         529
bundle  NORTH bundle:0003        1     2026-01-21 UPSERT R8e867287f9dc Rc2e7f1ec75e0         187
bundle  NORTH bundle:0001        2     2026-02-21 UPSERT R7308b473a5a8 Rf6f2a500079e         129
bundle  NORTH bundle:0007        2     2026-02-21 UPSERT R148f899e57f6 Rc07011e6ab2d         269
bundle  NORTH bundle:0003        2     2026-02-21 UPSERT R8e867287f9dc Rc2e7f1ec75e0         180
bundle  NORTH bundle:0007        2     2026-02-21 UPSERT R148f899e57f6 Rc07011e6ab2d         269
bundle  NORTH bundle:0005        1     2026-01-21 UPSERT R698808ad9a96 R07e87ed82812         517
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 5}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 4}, "item_b": {"missing": 0, "unique_nonempty": 5}, "bonus_cents": {"missing": 0, "unique_nonempty": 10, "numeric_range": [129.0, 529.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "bundle", "matching_columns": [{"column": "table", "exact": 11, "prefix": 11, "contains": 11, "examples": ["bundle"]}, {"column": "record_id", "exact": 0, "prefix": 11, "contains": 11, "examples": ["bundle:0001", "bundle:0000", "bundle:0007"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 10, "prefix": 10, "contains": 10, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_05/export_05.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 58
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'component', 'amount_cents']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'component': 'object', 'amount_cents': 'int64'}
Preview only (first 10 rows):
  table tenant    record_id revision effective_date action      item_ref     component amount_cents
benefit  NORTH benefit:0012        1     2026-01-21 UPSERT R148f899e57f6    item6_base          680
benefit  NORTH benefit:0027        1     2026-01-21 UPSERT R747b4b07f2df item13_rebate          -20
benefit  NORTH benefit:0024        1     2026-01-21 UPSERT R665d41e9e962   item12_base          968
benefit  NORTH benefit:0063        2     2026-02-21 UPSERT R07e87ed82812 item31_rebate          -18
benefit  NORTH benefit:0036        2     2026-02-21 UPSERT Rfb39090aa5c5   item18_base          684
benefit  NORTH benefit:0054        2     2026-02-21 UPSERT Rf9db851ed8bb   item27_base         9034
benefit  NORTH benefit:0039        1     2026-01-21 UPSERT R698808ad9a96 item19_rebate          -37
benefit  NORTH benefit:0045        2     2026-02-21 UPSERT Rc07011e6ab2d item22_rebate           -5
benefit  NORTH benefit:0042        1     2026-01-21 UPSERT Rd05f0610c770   item21_base          420
benefit  NORTH benefit:0018        1     2026-01-21 UPSERT R1b36613ee493    item9_base         1043
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 27}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 27}, "component": {"missing": 0, "unique_nonempty": 27}, "amount_cents": {"missing": 0, "unique_nonempty": 41, "numeric_range": [-44.0, 9040.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "benefit", "matching_columns": [{"column": "table", "exact": 58, "prefix": 58, "contains": 58, "examples": ["benefit"]}, {"column": "record_id", "exact": 0, "prefix": 58, "contains": 58, "examples": ["benefit:0012", "benefit:0027", "benefit:0024"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 53, "prefix": 53, "contains": 53, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_06/export_06.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 83
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
table tenant  record_id revision effective_date action      item_ref resource amount  unit
usage  NORTH usage:0040        2     2026-02-21 UPSERT R747b4b07f2df    labor      8  hour
usage  NORTH usage:0028        2     2026-02-21 UPSERT R1b36613ee493    labor      6  hour
usage  SOUTH usage:0012        2     2026-02-21 UPSERT R65d8546a7fb6    space      6 liter
usage  NORTH usage:0082        1     2026-01-21 UPSERT Rf9db851ed8bb    labor      9  hour
usage  NORTH usage:0082        2     2026-02-21 UPSERT Rf9db851ed8bb    labor      2  hour
usage  NORTH usage:0001        1     2026-01-21 UPSERT R6007124a9f06    labor     12  hour
usage  NORTH usage:0040        1     2026-01-21 UPSERT R747b4b07f2df    labor     15  hour
usage  NORTH usage:0031        1     2026-01-21 UPSERT R2b2709069f4c    labor     10  hour
usage  NORTH usage:0019        1     2026-01-21 UPSERT R148f899e57f6    labor     12  hour
usage  NORTH usage:0046        1     2026-01-21 UPSERT Re7d589eb4737    labor      9  hour
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 40}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 32}, "resource": {"missing": 0, "unique_nonempty": 2}, "amount": {"missing": 0, "unique_nonempty": 20, "numeric_range": [2.0, 27.0]}, "unit": {"missing": 0, "unique_nonempty": 2}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "hour", "matching_columns": [{"column": "unit", "exact": 75, "prefix": 75, "contains": 75, "examples": ["hour"]}], "exact_matching_columns": 1}, {"term": "liter", "matching_columns": [{"column": "unit", "exact": 8, "prefix": 8, "contains": 8, "examples": ["liter"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 75, "prefix": 75, "contains": 75, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 83, "prefix": 83, "contains": 83, "examples": ["usage"]}, {"column": "record_id", "exact": 0, "prefix": 83, "contains": 83, "examples": ["usage:0040", "usage:0028", "usage:0012"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_01/export_07.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 14
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_a', 'item_b']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_a': 'object', 'item_b': 'object'}
Preview only (first 10 rows):
       table tenant         record_id revision effective_date action        item_a        item_b
incompatible  SOUTH incompatible:0000        2     2026-02-21 UPSERT R65d8546a7fb6 R665d41e9e962
incompatible  NORTH incompatible:0009        2     2026-02-21 UPSERT R7955a6f70abc R698808ad9a96
incompatible  NORTH incompatible:0001        2     2026-02-21 UPSERT R747b4b07f2df Rf6f2a500079e
incompatible  NORTH incompatible:0001        1     2026-01-21 UPSERT R747b4b07f2df Rf6f2a500079e
incompatible  SOUTH incompatible:0008        2     2026-02-21 UPSERT Rfb39090aa5c5 R148f899e57f6
incompatible  NORTH incompatible:0007        1     2026-01-21 UPSERT R7308b473a5a8 Rf9db851ed8bb
incompatible  NORTH incompatible:0005        2     2026-02-21 UPSERT R747b4b07f2df Rd05f0610c770
incompatible  NORTH incompatible:0003        2     2026-02-21 UPSERT R9197b70e3d09 R665d41e9e962
incompatible  NORTH incompatible:0007        2     2026-02-21 UPSERT R7308b473a5a8 Rf9db851ed8bb
incompatible  NORTH incompatible:0009        1     2026-01-21 UPSERT R7955a6f70abc R698808ad9a96
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 7}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_a": {"missing": 0, "unique_nonempty": 6}, "item_b": {"missing": 0, "unique_nonempty": 6}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "incompatible", "matching_columns": [{"column": "table", "exact": 14, "prefix": 14, "contains": 14, "examples": ["incompatible"]}, {"column": "record_id", "exact": 0, "prefix": 14, "contains": 14, "examples": ["incompatible:0000", "incompatible:0009", "incompatible:0001"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 12, "prefix": 12, "contains": 12, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_02/export_08.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 11
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'resource', 'entry', 'amount', 'unit']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'resource': 'object', 'entry': 'object', 'amount': 'float64', 'unit': 'object'}
Preview only (first 10 rows):
          table tenant                 record_id revision effective_date action resource   entry amount   unit
capacity_ledger  NORTH      capacity_ledger:0004        1     2026-01-21 UPSERT    power opening 246007     wh
capacity_ledger  NORTH      capacity_ledger:0002        2     2026-02-21 UPSERT    labor opening  13140 minute
capacity_ledger  NORTH      capacity_ledger:0000        2     2026-02-21 UPSERT    space opening 241000     ml
capacity_ledger  NORTH      capacity_ledger:0002        1     2026-01-21 UPSERT    labor opening  13147 minute
capacity_ledger  NORTH      capacity_ledger:0000        3     2026-03-23 UPSERT    space opening 241019     ml
capacity_ledger  NORTH capacity_ledger:withdrawn        2     2026-03-02 DELETE                               
capacity_ledger  NORTH      capacity_ledger:0004        2     2026-02-21 UPSERT    power opening 246000     wh
capacity_ledger  NORTH capacity_ledger:withdrawn        1     2026-01-21 UPSERT    space opening 241000     ml
capacity_ledger  NORTH      capacity_ledger:0000        2     2026-02-21 UPSERT    space opening 241000     ml
capacity_ledger  NORTH      capacity_ledger:0000        1     2026-01-21 UPSERT    space opening 241007     ml
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 4}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "resource": {"missing": 1, "unique_nonempty": 3}, "entry": {"missing": 1, "unique_nonempty": 1}, "amount": {"missing": 1, "unique_nonempty": 7, "numeric_range": [13140.0, 246007.0]}, "unit": {"missing": 1, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity_ledger", "matching_columns": [{"column": "table", "exact": 11, "prefix": 11, "contains": 11, "examples": ["capacity_ledger"]}, {"column": "record_id", "exact": 0, "prefix": 11, "contains": 11, "examples": ["capacity_ledger:0004", "capacity_ledger:0002", "capacity_ledger:0000"]}], "exact_matching_columns": 1}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "ml", "matching_columns": [{"column": "unit", "exact": 5, "prefix": 5, "contains": 5, "examples": ["ml"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 10, "prefix": 10, "contains": 10, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "wh", "matching_columns": [{"column": "unit", "exact": 3, "prefix": 3, "contains": 3, "examples": ["wh"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_03/export_09.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 41
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'historical_sales', 'forecast_units', 'margin_percent']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'historical_sales': 'int64', 'forecast_units': 'int64', 'margin_percent': 'int64'}
Preview only (first 10 rows):
 table tenant   record_id revision effective_date action      item_ref historical_sales forecast_units margin_percent
market  NORTH market:0001        2     2026-02-21 UPSERT Ref851ccaa158                3              2             28
market  NORTH market:0003        2     2026-02-21 UPSERT R7955a6f70abc                2              2             26
market  NORTH market:0003        1     2026-01-21 UPSERT R7955a6f70abc                9              9             33
market  NORTH market:0009        1     2026-01-21 UPSERT R1b36613ee493               10             10             17
market  NORTH market:0015        1     2026-01-21 UPSERT Re7d589eb4737                9              9             42
market  NORTH market:0007        2     2026-02-21 UPSERT R1820116348cd                3              1             18
market  SOUTH market:0008        2     2026-02-21 UPSERT Rf6f2a500079e                1              2             31
market  NORTH market:0013        1     2026-01-21 UPSERT R747b4b07f2df                9             10             41
market  NORTH market:0017        1     2026-01-21 UPSERT R0f58f5015d0d               10              9             36
market  NORTH market:0023        2     2026-02-21 UPSERT Ra5833cedea5c                3              2             16
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 20}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 20}, "historical_sales": {"missing": 0, "unique_nonempty": 8, "numeric_range": [1.0, 22.0]}, "forecast_units": {"missing": 0, "unique_nonempty": 9, "numeric_range": [1.0, 22.0]}, "margin_percent": {"missing": 0, "unique_nonempty": 30, "numeric_range": [5.0, 54.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "north", "matching_columns": [{"column": "tenant", "exact": 37, "prefix": 37, "contains": 37, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_04/export_10.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 45
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'category': 'object', 'authorized': 'float64', 'minimum_lot': 'float64', 'maximum_order': 'float64'}
Preview only (first 10 rows):
table tenant record_id revision effective_date action      item_ref category authorized minimum_lot maximum_order
 item  NORTH item:0018        1     2026-01-21 UPSERT Rfb39090aa5c5       G2          8           9            20
 item  NORTH item:0012        2     2026-02-21 UPSERT R665d41e9e962       G0          1           2             8
 item  NORTH item:0016        1     2026-01-21 UPSERT R3a11be59ddaa       G0          8           9            17
 item  NORTH item:0006        1     2026-01-21 UPSERT R148f899e57f6       G2          8           9            20
 item  SOUTH item:0020        2     2026-02-21 UPSERT Rc2e7f1ec75e0       G0          1           2             7
 item  NORTH item:0008        2     2026-02-21 UPSERT Rf6f2a500079e       G0          1           2            13
 item  NORTH item:0016        2     2026-02-21 UPSERT R3a11be59ddaa       G0          1           2            10
 item  NORTH item:0008        1     2026-01-21 UPSERT Rf6f2a500079e       G0          8           9            20
 item  NORTH item:0028        2     2026-02-21 UPSERT R7dcc05e5ca15       G0          1           2            11
 item  NORTH item:0022        2     2026-02-21 UPSERT Rc07011e6ab2d       G2          1           2             9
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 17}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_ref": {"missing": 1, "unique_nonempty": 16}, "category": {"missing": 1, "unique_nonempty": 2}, "authorized": {"missing": 1, "unique_nonempty": 5, "numeric_range": [0.0, 20.0]}, "minimum_lot": {"missing": 1, "unique_nonempty": 3, "numeric_range": [2.0, 21.0]}, "maximum_order": {"missing": 1, "unique_nonempty": 17, "numeric_range": [7.0, 30.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "authorized", "matching_columns": [], "exact_matching_columns": 0}, {"term": "category", "matching_columns": [], "exact_matching_columns": 0}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "item", "matching_columns": [{"column": "table", "exact": 45, "prefix": 45, "contains": 45, "examples": ["item"]}, {"column": "record_id", "exact": 0, "prefix": 45, "contains": 45, "examples": ["item:0018", "item:0012", "item:0016"]}], "exact_matching_columns": 1}, {"term": "maximum_order", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 41, "prefix": 41, "contains": 41, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_05/export_11.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 13
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_a', 'item_b', 'bonus_cents']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_a': 'object', 'item_b': 'object', 'bonus_cents': 'float64'}
Preview only (first 10 rows):
 table tenant        record_id revision effective_date action        item_a        item_b bonus_cents
bundle  NORTH      bundle:0002        2     2026-02-21 UPSERT R7308b473a5a8 R1218064667a2         350
bundle  NORTH      bundle:0006        1     2026-01-21 UPSERT Rd05f0610c770 R7308b473a5a8         230
bundle  NORTH bundle:withdrawn        1     2026-01-21 UPSERT R148f899e57f6 R698808ad9a96         174
bundle  NORTH      bundle:0004        1     2026-01-21 UPSERT Rc2e7f1ec75e0 R0f58f5015d0d         557
bundle  NORTH      bundle:0000        1     2026-01-21 UPSERT R148f899e57f6 R698808ad9a96         181
bundle  NORTH      bundle:0004        2     2026-02-21 UPSERT Rc2e7f1ec75e0 R0f58f5015d0d         550
bundle  NORTH      bundle:0006        2     2026-02-21 UPSERT Rd05f0610c770 R7308b473a5a8         223
bundle  NORTH      bundle:0002        1     2026-01-21 UPSERT R7308b473a5a8 R1218064667a2         357
bundle  NORTH      bundle:0000        2     2026-02-21 UPSERT R148f899e57f6 R698808ad9a96         174
bundle  SOUTH      bundle:0004        2     2026-02-21 UPSERT Rc2e7f1ec75e0 R0f58f5015d0d         550
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 5}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_a": {"missing": 1, "unique_nonempty": 4}, "item_b": {"missing": 1, "unique_nonempty": 4}, "bonus_cents": {"missing": 1, "unique_nonempty": 9, "numeric_range": [174.0, 557.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "bundle", "matching_columns": [{"column": "table", "exact": 13, "prefix": 13, "contains": 13, "examples": ["bundle"]}, {"column": "record_id", "exact": 0, "prefix": 13, "contains": 13, "examples": ["bundle:0002", "bundle:0006", "bundle:withdrawn"]}], "exact_matching_columns": 1}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 12, "prefix": 12, "contains": 12, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_06/export_12.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 56
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'component', 'amount_cents']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'component': 'object', 'amount_cents': 'float64'}
Preview only (first 10 rows):
  table tenant    record_id revision effective_date action      item_ref     component amount_cents
benefit  NORTH benefit:0004        2     2026-02-21 UPSERT R1218064667a2    item2_base         1120
benefit  NORTH benefit:0025        3     2026-03-23 UPSERT R665d41e9e962 item12_rebate           -4
benefit  NORTH benefit:0013        1     2026-01-21 UPSERT R148f899e57f6  item6_rebate          -23
benefit  NORTH benefit:0055        1     2026-01-21 UPSERT Rf9db851ed8bb item27_rebate          -34
benefit  NORTH benefit:0028        2     2026-02-21 UPSERT R8e867287f9dc   item14_base         1186
benefit  NORTH benefit:0049        2     2026-02-21 UPSERT Rcf5370ab2e7c item24_rebate          -33
benefit  NORTH benefit:0001        1     2026-01-21 UPSERT R6007124a9f06  item0_rebate          -26
benefit  NORTH benefit:0010        1     2026-01-21 UPSERT Rc9f004c5ca0f    item5_base         1213
benefit  NORTH benefit:0061        1     2026-01-21 UPSERT R30896725a0ac item30_rebate          -37
benefit  NORTH benefit:0016        2     2026-02-21 UPSERT Rf6f2a500079e    item8_base         1215
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 27}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_ref": {"missing": 1, "unique_nonempty": 26}, "component": {"missing": 1, "unique_nonempty": 27}, "amount_cents": {"missing": 1, "unique_nonempty": 39, "numeric_range": [-49.0, 9038.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "benefit", "matching_columns": [{"column": "table", "exact": 56, "prefix": 56, "contains": 56, "examples": ["benefit"]}, {"column": "record_id", "exact": 0, "prefix": 56, "contains": 56, "examples": ["benefit:0004", "benefit:0025", "benefit:0013"]}], "exact_matching_columns": 1}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 51, "prefix": 51, "contains": 51, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_01/export_13.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 41
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'category': 'object', 'authorized': 'int64', 'minimum_lot': 'int64', 'maximum_order': 'int64'}
Preview only (first 10 rows):
table tenant record_id revision effective_date action      item_ref category authorized minimum_lot maximum_order
 item  NORTH item:0001        1     2026-01-21 UPSERT Ref851ccaa158       G1          8           9            15
 item  NORTH item:0003        2     2026-02-21 UPSERT R7955a6f70abc       G3          1           2             7
 item  NORTH item:0003        1     2026-01-21 UPSERT R7955a6f70abc       G3          8           9            14
 item  NORTH item:0009        2     2026-02-21 UPSERT R1b36613ee493       G1          1           2             7
 item  NORTH item:0015        1     2026-01-21 UPSERT Re7d589eb4737       G3          7           9            19
 item  NORTH item:0007        2     2026-02-21 UPSERT R1820116348cd       G3          1           2            10
 item  SOUTH item:0008        2     2026-02-21 UPSERT Rf6f2a500079e       G0          1           2            13
 item  NORTH item:0013        2     2026-02-21 UPSERT R747b4b07f2df       G1          1           2            11
 item  NORTH item:0017        1     2026-01-21 UPSERT R0f58f5015d0d       G1          8           9            16
 item  NORTH item:0023        2     2026-02-21 UPSERT Ra5833cedea5c       G3          0           2            10
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 20}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 20}, "category": {"missing": 0, "unique_nonempty": 3}, "authorized": {"missing": 0, "unique_nonempty": 6, "numeric_range": [0.0, 20.0]}, "minimum_lot": {"missing": 0, "unique_nonempty": 3, "numeric_range": [2.0, 21.0]}, "maximum_order": {"missing": 0, "unique_nonempty": 17, "numeric_range": [7.0, 31.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "authorized", "matching_columns": [], "exact_matching_columns": 0}, {"term": "category", "matching_columns": [], "exact_matching_columns": 0}, {"term": "item", "matching_columns": [{"column": "table", "exact": 41, "prefix": 41, "contains": 41, "examples": ["item"]}, {"column": "record_id", "exact": 0, "prefix": 41, "contains": 41, "examples": ["item:0001", "item:0003", "item:0009"]}], "exact_matching_columns": 1}, {"term": "maximum_order", "matching_columns": [], "exact_matching_columns": 0}, {"term": "minimum_lot", "matching_columns": [], "exact_matching_columns": 0}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 37, "prefix": 37, "contains": 37, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_02/export_14.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 15
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_a', 'item_b']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_a': 'object', 'item_b': 'object'}
Preview only (first 10 rows):
       table tenant              record_id revision effective_date action        item_a        item_b
incompatible  SOUTH      incompatible:0004        2     2026-02-21 UPSERT R1b36613ee493 R8e867287f9dc
incompatible  NORTH      incompatible:0006        1     2026-01-21 UPSERT R9197b70e3d09 R7308b473a5a8
incompatible  NORTH      incompatible:0002        1     2026-01-21 UPSERT Rf9db851ed8bb R747b4b07f2df
incompatible  NORTH      incompatible:0008        2     2026-02-21 UPSERT Rfb39090aa5c5 R148f899e57f6
incompatible  NORTH incompatible:withdrawn        2     2026-03-02 DELETE                            
incompatible  NORTH      incompatible:0004        1     2026-01-21 UPSERT R1b36613ee493 R8e867287f9dc
incompatible  NORTH      incompatible:0000        2     2026-02-21 UPSERT R65d8546a7fb6 R665d41e9e962
incompatible  NORTH      incompatible:0006        2     2026-02-21 UPSERT R9197b70e3d09 R7308b473a5a8
incompatible  NORTH      incompatible:0004        2     2026-02-21 UPSERT R1b36613ee493 R8e867287f9dc
incompatible  NORTH      incompatible:0002        2     2026-02-21 UPSERT Rf9db851ed8bb R747b4b07f2df
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 6}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_a": {"missing": 1, "unique_nonempty": 5}, "item_b": {"missing": 1, "unique_nonempty": 5}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "incompatible", "matching_columns": [{"column": "table", "exact": 15, "prefix": 15, "contains": 15, "examples": ["incompatible"]}, {"column": "record_id", "exact": 0, "prefix": 15, "contains": 15, "examples": ["incompatible:0004", "incompatible:0006", "incompatible:0002"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 14, "prefix": 14, "contains": 14, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_03/export_15.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 55
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'component', 'amount_cents']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'component': 'object', 'amount_cents': 'int64'}
Preview only (first 10 rows):
  table tenant    record_id revision effective_date action      item_ref     component amount_cents
benefit  NORTH benefit:0008        2     2026-02-21 UPSERT R65d8546a7fb6    item4_base         1301
benefit  NORTH benefit:0029        1     2026-01-21 UPSERT R8e867287f9dc item14_rebate          -22
benefit  NORTH benefit:0047        1     2026-01-21 UPSERT Ra5833cedea5c item23_rebate          -28
benefit  NORTH benefit:0035        2     2026-02-21 UPSERT R0f58f5015d0d item17_rebate          -24
benefit  NORTH benefit:0020        3     2026-03-23 UPSERT R2b2709069f4c   item10_base          420
benefit  NORTH benefit:0041        1     2026-01-21 UPSERT Rc2e7f1ec75e0 item20_rebate           -7
benefit  NORTH benefit:0047        2     2026-02-21 UPSERT Ra5833cedea5c item23_rebate          -31
benefit  NORTH benefit:0014        1     2026-01-21 UPSERT R1820116348cd    item7_base          504
benefit  NORTH benefit:0014        2     2026-02-21 UPSERT R1820116348cd    item7_base          504
benefit  NORTH benefit:0026        1     2026-01-21 UPSERT R747b4b07f2df   item13_base         1622
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 27}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 27}, "component": {"missing": 0, "unique_nonempty": 27}, "amount_cents": {"missing": 0, "unique_nonempty": 39, "numeric_range": [-50.0, 9033.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "amount_cents", "matching_columns": [], "exact_matching_columns": 0}, {"term": "benefit", "matching_columns": [{"column": "table", "exact": 55, "prefix": 55, "contains": 55, "examples": ["benefit"]}, {"column": "record_id", "exact": 0, "prefix": 55, "contains": 55, "examples": ["benefit:0008", "benefit:0029", "benefit:0047"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 49, "prefix": 49, "contains": 49, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_04/export_16.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 5
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'category': 'object', 'minimum_quantity': 'int64', 'maximum_quantity': 'int64', 'activation_fee_cents': 'int64'}
Preview only (first 10 rows):
   table tenant     record_id revision effective_date action category minimum_quantity maximum_quantity activation_fee_cents
category  NORTH category:0001        1     2026-01-21 UPSERT       G1               12               24                  339
category  NORTH category:0001        2     2026-02-21 UPSERT       G1                5               17                  332
category  SOUTH category:0000        2     2026-02-21 UPSERT       G0                9               21                  162
category  NORTH category:0003        2     2026-02-21 UPSERT       G3                9               21                  166
category  NORTH category:0003        1     2026-01-21 UPSERT       G3               16               28                  173
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 3}, "revision": {"missing": 0, "unique_nonempty": 2, "numeric_range": [1.0, 2.0]}, "effective_date": {"missing": 0, "unique_nonempty": 2}, "action": {"missing": 0, "unique_nonempty": 1}, "category": {"missing": 0, "unique_nonempty": 3}, "minimum_quantity": {"missing": 0, "unique_nonempty": 4, "numeric_range": [5.0, 16.0]}, "maximum_quantity": {"missing": 0, "unique_nonempty": 4, "numeric_range": [17.0, 28.0]}, "activation_fee_cents": {"missing": 0, "unique_nonempty": 5, "numeric_range": [162.0, 339.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "category", "matching_columns": [{"column": "table", "exact": 5, "prefix": 5, "contains": 5, "examples": ["category"]}, {"column": "record_id", "exact": 0, "prefix": 5, "contains": 5, "examples": ["category:0001", "category:0000", "category:0003"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 4, "prefix": 4, "contains": 4, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_05/export_17.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 41
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'activation_fee_cents': 'int64'}
Preview only (first 10 rows):
   table tenant     record_id revision effective_date action      item_ref activation_fee_cents
item_fee  NORTH item_fee:0001        1     2026-01-21 UPSERT Ref851ccaa158                  214
item_fee  NORTH item_fee:0003        1     2026-01-21 UPSERT R7955a6f70abc                  317
item_fee  NORTH item_fee:0003        2     2026-02-21 UPSERT R7955a6f70abc                  310
item_fee  NORTH item_fee:0009        1     2026-01-21 UPSERT R1b36613ee493                  144
item_fee  NORTH item_fee:0015        1     2026-01-21 UPSERT Re7d589eb4737                  202
item_fee  NORTH item_fee:0007        1     2026-01-21 UPSERT R1820116348cd                  464
item_fee  SOUTH item_fee:0008        2     2026-02-21 UPSERT Rf6f2a500079e                  217
item_fee  NORTH item_fee:0013        1     2026-01-21 UPSERT R747b4b07f2df                  351
item_fee  NORTH item_fee:0017        2     2026-02-21 UPSERT R0f58f5015d0d                  302
item_fee  NORTH item_fee:0023        2     2026-02-21 UPSERT Ra5833cedea5c                  466
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 20}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 20}, "activation_fee_cents": {"missing": 0, "unique_nonempty": 38, "numeric_range": [104.0, 502.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "item_fee", "matching_columns": [{"column": "table", "exact": 41, "prefix": 41, "contains": 41, "examples": ["item_fee"]}, {"column": "record_id", "exact": 0, "prefix": 41, "contains": 41, "examples": ["item_fee:0001", "item_fee:0003", "item_fee:0009"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 37, "prefix": 37, "contains": 37, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_06/export_18.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'category': 'object', 'minimum_quantity': 'float64', 'maximum_quantity': 'float64', 'activation_fee_cents': 'float64'}
Preview only (first 10 rows):
   table tenant          record_id revision effective_date action category minimum_quantity maximum_quantity activation_fee_cents
category  NORTH      category:0002        2     2026-02-21 UPSERT       G2                9               21                  199
category  NORTH      category:0000        1     2026-01-21 UPSERT       G0               16               28                  169
category  NORTH      category:0002        1     2026-01-21 UPSERT       G2               16               28                  206
category  NORTH category:withdrawn        2     2026-03-02 DELETE                                                                
category  NORTH      category:0000        2     2026-02-21 UPSERT       G0                9               21                  162
category  NORTH category:withdrawn        1     2026-01-21 UPSERT       G0                9               21                  162
category  NORTH      category:0000        3     2026-03-23 UPSERT       G0               28               40                  181
category  NORTH      category:0000        2     2026-02-21 UPSERT       G0                9               21                  162
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 1}, "record_id": {"missing": 0, "unique_nonempty": 3}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "category": {"missing": 1, "unique_nonempty": 2}, "minimum_quantity": {"missing": 1, "unique_nonempty": 3, "numeric_range": [9.0, 28.0]}, "maximum_quantity": {"missing": 1, "unique_nonempty": 3, "numeric_range": [21.0, 40.0]}, "activation_fee_cents": {"missing": 1, "unique_nonempty": 5, "numeric_range": [162.0, 206.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "category", "matching_columns": [{"column": "table", "exact": 8, "prefix": 8, "contains": 8, "examples": ["category"]}, {"column": "record_id", "exact": 0, "prefix": 8, "contains": 8, "examples": ["category:0002", "category:0000", "category:withdrawn"]}], "exact_matching_columns": 1}, {"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 8, "prefix": 8, "contains": 8, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_01/export_19.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 83
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
table tenant  record_id revision effective_date action      item_ref resource amount unit
usage  NORTH usage:0071        1     2026-01-21 UPSERT Ra5833cedea5c    power     10  kwh
usage  NORTH usage:0065        1     2026-01-21 UPSERT Rd05f0610c770    power     12  kwh
usage  NORTH usage:0056        1     2026-01-21 UPSERT Rfb39090aa5c5    power     15  kwh
usage  NORTH usage:0023        1     2026-01-21 UPSERT R1820116348cd    power     16  kwh
usage  NORTH usage:0092        1     2026-01-21 UPSERT R30896725a0ac    power     16  kwh
usage  NORTH usage:0014        2     2026-02-21 UPSERT R65d8546a7fb6    power      3  kwh
usage  NORTH usage:0011        1     2026-01-21 UPSERT R7955a6f70abc    power      9  kwh
usage  NORTH usage:0038        2     2026-02-21 UPSERT R665d41e9e962    power      2  kwh
usage  SOUTH usage:0040        2     2026-02-21 UPSERT R747b4b07f2df    labor      8 hour
usage  NORTH usage:0020        2     2026-02-21 UPSERT R148f899e57f6    power      5  kwh
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 40}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "item_ref": {"missing": 0, "unique_nonempty": 32}, "resource": {"missing": 0, "unique_nonempty": 2}, "amount": {"missing": 0, "unique_nonempty": 19, "numeric_range": [2.0, 27.0]}, "unit": {"missing": 0, "unique_nonempty": 2}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "hour", "matching_columns": [{"column": "unit", "exact": 8, "prefix": 8, "contains": 8, "examples": ["hour"]}], "exact_matching_columns": 1}, {"term": "kwh", "matching_columns": [{"column": "unit", "exact": 75, "prefix": 75, "contains": 75, "examples": ["kwh"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 75, "prefix": 75, "contains": 75, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 83, "prefix": 83, "contains": 83, "examples": ["usage"]}, {"column": "record_id", "exact": 0, "prefix": 83, "contains": 83, "examples": ["usage:0071", "usage:0065", "usage:0056"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_02/export_20.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 8
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'resource', 'entry', 'amount', 'unit']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'resource': 'object', 'entry': 'object', 'amount': 'int64', 'unit': 'object'}
Preview only (first 10 rows):
          table tenant            record_id revision effective_date action resource       entry amount   unit
capacity_ledger  NORTH capacity_ledger:0003        1     2026-01-21 UPSERT    labor reservation   -293 minute
capacity_ledger  NORTH capacity_ledger:0001        2     2026-02-21 UPSERT    space reservation  -6000     ml
capacity_ledger  NORTH capacity_ledger:0005        2     2026-02-21 UPSERT    power reservation -11000     wh
capacity_ledger  NORTH capacity_ledger:0005        3     2026-03-23 UPSERT    power reservation -10981     wh
capacity_ledger  NORTH capacity_ledger:0001        1     2026-01-21 UPSERT    space reservation  -5993     ml
capacity_ledger  SOUTH capacity_ledger:0000        2     2026-02-21 UPSERT    space     opening 241000     ml
capacity_ledger  NORTH capacity_ledger:0003        2     2026-02-21 UPSERT    labor reservation   -300 minute
capacity_ledger  NORTH capacity_ledger:0005        1     2026-01-21 UPSERT    power reservation -10993     wh
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 4}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 3}, "action": {"missing": 0, "unique_nonempty": 1}, "resource": {"missing": 0, "unique_nonempty": 3}, "entry": {"missing": 0, "unique_nonempty": 2}, "amount": {"missing": 0, "unique_nonempty": 8, "numeric_range": [-11000.0, 241000.0]}, "unit": {"missing": 0, "unique_nonempty": 3}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "capacity_ledger", "matching_columns": [{"column": "table", "exact": 8, "prefix": 8, "contains": 8, "examples": ["capacity_ledger"]}, {"column": "record_id", "exact": 0, "prefix": 8, "contains": 8, "examples": ["capacity_ledger:0003", "capacity_ledger:0001", "capacity_ledger:0005"]}], "exact_matching_columns": 1}, {"term": "ml", "matching_columns": [{"column": "unit", "exact": 3, "prefix": 3, "contains": 3, "examples": ["ml"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 7, "prefix": 7, "contains": 7, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "wh", "matching_columns": [{"column": "unit", "exact": 3, "prefix": 3, "contains": 3, "examples": ["wh"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_03/export_21.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 45
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'historical_sales', 'forecast_units', 'margin_percent']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'historical_sales': 'float64', 'forecast_units': 'float64', 'margin_percent': 'float64'}
Preview only (first 10 rows):
 table tenant   record_id revision effective_date action      item_ref historical_sales forecast_units margin_percent
market  NORTH market:0018        2     2026-02-21 UPSERT Rfb39090aa5c5                1              2             28
market  NORTH market:0012        1     2026-01-21 UPSERT R665d41e9e962                9              8             12
market  NORTH market:0016        2     2026-02-21 UPSERT R3a11be59ddaa                3              2             15
market  NORTH market:0006        1     2026-01-21 UPSERT R148f899e57f6               10              8             15
market  SOUTH market:0020        2     2026-02-21 UPSERT Rc2e7f1ec75e0                2              2             25
market  NORTH market:0008        1     2026-01-21 UPSERT Rf6f2a500079e                8              9             38
market  NORTH market:0016        1     2026-01-21 UPSERT R3a11be59ddaa               10              9             22
market  NORTH market:0008        2     2026-02-21 UPSERT Rf6f2a500079e                1              2             31
market  NORTH market:0028        2     2026-02-21 UPSERT R7dcc05e5ca15                2              2             33
market  NORTH market:0022        2     2026-02-21 UPSERT Rc07011e6ab2d                3              2             18
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 17}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_ref": {"missing": 1, "unique_nonempty": 16}, "historical_sales": {"missing": 1, "unique_nonempty": 8, "numeric_range": [1.0, 22.0]}, "forecast_units": {"missing": 1, "unique_nonempty": 9, "numeric_range": [1.0, 22.0]}, "margin_percent": {"missing": 1, "unique_nonempty": 26, "numeric_range": [5.0, 54.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 41, "prefix": 41, "contains": 41, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_04/export_22.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 45
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'ref', 'kind', 'entity_id']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'ref': 'object', 'kind': 'object', 'entity_id': 'object'}
Preview only (first 10 rows):
   table tenant     record_id revision effective_date action           ref kind entity_id
identity  NORTH identity:0018        2     2026-02-21 UPSERT Rfb39090aa5c5 item       I18
identity  NORTH identity:0012        1     2026-01-21 UPSERT R665d41e9e962 item       I12
identity  NORTH identity:0016        2     2026-02-21 UPSERT R3a11be59ddaa item       I16
identity  NORTH identity:0006        2     2026-02-21 UPSERT R148f899e57f6 item       I06
identity  SOUTH identity:0020        2     2026-02-21 UPSERT Rc2e7f1ec75e0 item       I20
identity  NORTH identity:0008        2     2026-02-21 UPSERT Rf6f2a500079e item       I08
identity  NORTH identity:0016        1     2026-01-21 UPSERT R3a11be59ddaa item       I16
identity  NORTH identity:0008        1     2026-01-21 UPSERT Rf6f2a500079e item       I08
identity  NORTH identity:0028        1     2026-01-21 UPSERT R7dcc05e5ca15 item       I28
identity  NORTH identity:0022        2     2026-02-21 UPSERT Rc07011e6ab2d item       I22
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 17}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "ref": {"missing": 1, "unique_nonempty": 16}, "kind": {"missing": 1, "unique_nonempty": 1}, "entity_id": {"missing": 1, "unique_nonempty": 16}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "item", "matching_columns": [{"column": "kind", "exact": 44, "prefix": 44, "contains": 44, "examples": ["item"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 41, "prefix": 41, "contains": 41, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_05/export_23.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 86
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'resource', 'amount', 'unit']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'resource': 'object', 'amount': 'float64', 'unit': 'object'}
Preview only (first 10 rows):
table tenant  record_id revision effective_date action      item_ref resource amount  unit
usage  NORTH usage:0003        1     2026-01-21 UPSERT Ref851ccaa158    space      9 liter
usage  NORTH usage:0069        1     2026-01-21 UPSERT Ra5833cedea5c    space     13 liter
usage  NORTH usage:0045        1     2026-01-21 UPSERT Re7d589eb4737    space     16 liter
usage  NORTH usage:0045        2     2026-02-21 UPSERT Re7d589eb4737    space      9 liter
usage  NORTH usage:0024        1     2026-01-21 UPSERT Rf6f2a500079e    space     12 liter
usage  NORTH usage:0054        1     2026-01-21 UPSERT Rfb39090aa5c5    space     16 liter
usage  NORTH usage:0048        2     2026-02-21 UPSERT R3a11be59ddaa    space      5 liter
usage  SOUTH usage:0056        2     2026-02-21 UPSERT Rfb39090aa5c5    power      8   kwh
usage  NORTH usage:0066        2     2026-02-21 UPSERT Rc07011e6ab2d    space      3 liter
usage  NORTH usage:0087        2     2026-02-21 UPSERT R9197b70e3d09    space      8 liter
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 41}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_ref": {"missing": 1, "unique_nonempty": 32}, "resource": {"missing": 1, "unique_nonempty": 2}, "amount": {"missing": 1, "unique_nonempty": 20, "numeric_range": [2.0, 28.0]}, "unit": {"missing": 1, "unique_nonempty": 2}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "kwh", "matching_columns": [{"column": "unit", "exact": 8, "prefix": 8, "contains": 8, "examples": ["kwh"]}], "exact_matching_columns": 1}, {"term": "liter", "matching_columns": [{"column": "unit", "exact": 77, "prefix": 77, "contains": 77, "examples": ["liter"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 78, "prefix": 78, "contains": 78, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "resource", "matching_columns": [], "exact_matching_columns": 0}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}, {"term": "unit", "matching_columns": [], "exact_matching_columns": 0}, {"term": "usage", "matching_columns": [{"column": "table", "exact": 86, "prefix": 86, "contains": 86, "examples": ["usage"]}, {"column": "record_id", "exact": 0, "prefix": 86, "contains": 86, "examples": ["usage:0003", "usage:0069", "usage:0045"]}], "exact_matching_columns": 1}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_06/export_24.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 28
Columns: ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'action', 'item_ref', 'prerequisite_ref']
Parsed column types: {'table': 'object', 'tenant': 'object', 'record_id': 'object', 'revision': 'int64', 'effective_date': 'object', 'action': 'object', 'item_ref': 'object', 'prerequisite_ref': 'object'}
Preview only (first 10 rows):
   table tenant          record_id revision effective_date action      item_ref prerequisite_ref
requires  NORTH      requires:0000        3     2026-03-23 UPSERT R665d41e9e962    Rc9f004c5ca0f
requires  NORTH      requires:0008        1     2026-01-21 UPSERT Rc2e7f1ec75e0    R65d8546a7fb6
requires  NORTH requires:withdrawn        2     2026-03-02 DELETE                               
requires  SOUTH      requires:0004        2     2026-02-21 UPSERT R3a11be59ddaa    R65d8546a7fb6
requires  NORTH      requires:0002        2     2026-02-21 UPSERT R8e867287f9dc    R1820116348cd
requires  NORTH      requires:0014        2     2026-02-21 UPSERT Rc6c554a79b27    Rf6f2a500079e
requires  NORTH      requires:0004        1     2026-01-21 UPSERT R3a11be59ddaa    R65d8546a7fb6
requires  NORTH      requires:0010        1     2026-01-21 UPSERT Rc07011e6ab2d    R6007124a9f06
requires  NORTH      requires:0006        1     2026-01-21 UPSERT Rfb39090aa5c5    R1820116348cd
requires  NORTH      requires:0010        2     2026-02-21 UPSERT Rc07011e6ab2d    R6007124a9f06
Full-file column statistics: {"table": {"missing": 0, "unique_nonempty": 1}, "tenant": {"missing": 0, "unique_nonempty": 2}, "record_id": {"missing": 0, "unique_nonempty": 11}, "revision": {"missing": 0, "unique_nonempty": 3, "numeric_range": [1.0, 3.0]}, "effective_date": {"missing": 0, "unique_nonempty": 4}, "action": {"missing": 0, "unique_nonempty": 2}, "item_ref": {"missing": 1, "unique_nonempty": 10}, "prerequisite_ref": {"missing": 1, "unique_nonempty": 8}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "delete", "matching_columns": [{"column": "action", "exact": 1, "prefix": 1, "contains": 1, "examples": ["DELETE"]}], "exact_matching_columns": 1}, {"term": "north", "matching_columns": [{"column": "tenant", "exact": 26, "prefix": 26, "contains": 26, "examples": ["NORTH"]}], "exact_matching_columns": 1}, {"term": "record_id", "matching_columns": [], "exact_matching_columns": 0}, {"term": "requires", "matching_columns": [{"column": "table", "exact": 28, "prefix": 28, "contains": 28, "examples": ["requires"]}, {"column": "record_id", "exact": 0, "prefix": 28, "contains": 28, "examples": ["requires:0000", "requires:0008", "requires:withdrawn"]}], "exact_matching_columns": 1}, {"term": "revision", "matching_columns": [], "exact_matching_columns": 0}, {"term": "table", "matching_columns": [], "exact_matching_columns": 0}, {"term": "tenant", "matching_columns": [], "exact_matching_columns": 0}]