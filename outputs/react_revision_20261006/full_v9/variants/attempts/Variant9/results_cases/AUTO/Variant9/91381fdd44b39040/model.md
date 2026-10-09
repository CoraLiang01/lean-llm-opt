## Integer Cutting-Stock Pattern-Selection Model

**Sets**
- $I$: set of item types (from item_demand.csv), $I = \{\text{A}, \text{B}, \text{C}, \text{D}\}$
- $P$: set of cutting patterns (from cutting_patterns.csv), $P = \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}, \text{P7}, \text{P8}, \text{P9}\}$

**Parameters**
- $d_i$: demand for item $i \in I$ (from item_demand.csv, column Demand, table_id file_0_view_0)
- $a_{pi}$: number of units of item $i$ produced by pattern $p$ (from cutting_patterns.csv, columns A,B,C,D, table_id file_1_view_0)

**Decision Variables**
- $y_p \in \mathbb{Z}_+, \quad \forall p \in P$: number of standard rolls cut using pattern $p$

**Objective**
\[
\min \sum_{p \in P} y_p
\]

**Constraints**
\[
\sum_{p \in P} a_{pi} y_p \geq d_i, \quad \forall i \in I
\]
\[
y_p \geq 0 \text{ and integer}, \quad \forall p \in P
\]

---

### Data Mapping

- $I$ (item types): file_0_view_0, column "Item"
- $P$ (patterns): file_1_view_0, column "Pattern"
- $d_i$: file_0_view_0, column "Demand", indexed by "Item"
- $a_{pi}$: file_1_view_0, columns "A", "B", "C", "D", indexed by "Pattern" and "Item"
- $y_p$: decision variable for each $p \in P$