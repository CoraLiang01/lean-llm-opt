## Symbolic Mathematical Model

**Sets**
- $I$: set of item types (from item_demand.csv), indexed by $i$
- $P$: set of cutting patterns (from cutting_patterns.csv), indexed by $p$

**Parameters**
- $d_i$: demand for item $i$ (from item_demand.csv, column "Demand", table_id: file_0_view_0)
- $a_{ip}$: number of units of item $i$ produced by pattern $p$ (from cutting_patterns.csv, columns "A", "B", "C", "D", table_id: file_1_view_0)

**Decision Variables**
- $y_p \in \mathbb{Z}_+, \ \forall p \in P$: number of standard rolls cut using pattern $p$

**Objective**
\[
\min \sum_{p \in P} y_p
\]

**Constraints**
\[
\sum_{p \in P} a_{ip} \, y_p \geq d_i \qquad \forall i \in I
\]
\[
y_p \geq 0 \text{ and integer} \qquad \forall p \in P
\]

---

## Data Mapping

- $I$ (item types): All "Item" values in item_demand.csv (table_id: file_0_view_0)
- $P$ (patterns): All "Pattern" values in cutting_patterns.csv (table_id: file_1_view_0)
- $d_i$: "Demand" for item $i$ in item_demand.csv (table_id: file_0_view_0, columns "Item", "Demand")
- $a_{ip}$: entry in cutting_patterns.csv for item $i$ and pattern $p$ (table_id: file_1_view_0, columns "Pattern", "A", "B", "C", "D")
- $y_p$: integer variable for each pattern $p$ (decision variable, not in data)

All sets, parameters, and constraints are defined directly from the current CSV data.