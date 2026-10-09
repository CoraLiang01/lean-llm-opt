## Mathematical Model

**Sets**
- $I$: set of items, from item_demand.csv (Item column)
- $P$: set of cutting patterns, from cutting_patterns.csv (Pattern column)

**Parameters**
- $d_i$: demand for item $i \in I$ (item_demand.csv, Demand column)
- $a_{ip}$: number of pieces of item $i$ produced by pattern $p$ (cutting_patterns.csv, columns A–E for each pattern)

**Decision Variables**
- $y_p \in \mathbb{Z}_+, \ \forall p \in P$: number of times pattern $p$ is used (number of rolls cut with pattern $p$)

**Objective**
\[
\min \sum_{p \in P} y_p
\]

**Constraints**
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

- $I$ (items): All Item values in table_id=file_0_view_0, column=Item
- $P$ (patterns): All Pattern values in table_id=file_1_view_0, column=Pattern
- $d_i$: Demand for item $i$ from table_id=file_0_view_0, column=Demand
- $a_{ip}$: Number of pieces of item $i$ in pattern $p$ from table_id=file_1_view_0, row with Pattern=$p$, column $i$ (A–E)
- $y_p$: Decision variable for each $p$ in $P$ (number of rolls cut with pattern $p$)

**All variables are nonnegative integers.**