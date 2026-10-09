## Cutting-Stock Integer Programming Model

**Sets:**
- $I$: set of items, from item_demand.csv (Item column)
- $P$: set of cutting patterns, from cutting_patterns.csv (Pattern column)

**Parameters:**
- $d_i$: demand for item $i \in I$ (item_demand.csv, Demand)
- $a_{ip}$: number of pieces of item $i$ produced by pattern $p$ (cutting_patterns.csv, columns A–E)
- All indices and values as in the source files.

**Decision Variables:**
- $y_p \in \mathbb{Z}_+, \ \forall p \in P$: number of times pattern $p$ is used (number of rolls cut with pattern $p$)

**Objective:**
\[
\min \sum_{p \in P} y_p
\]

**Constraints:**
- Demand satisfaction for each item:
\[
\sum_{p \in P} a_{ip} \, y_p \geq d_i \qquad \forall i \in I
\]
- Nonnegativity and integrality:
\[
y_p \in \mathbb{Z}_+, \qquad \forall p \in P
\]

---

### Data Mapping

- $I$ (items): file_0_view_0, column "Item"
- $d_i$: file_0_view_0, column "Demand", indexed by "Item"
- $P$ (patterns): file_1_view_0, column "Pattern"
- $a_{ip}$: file_1_view_0, columns "A", "B", "C", "D", "E", indexed by ("Pattern", "Item")
- $y_p$: decision variable for each $p$ in $P$ (Pattern)

All sets, parameters, and constraints are defined directly from the current CSV data.