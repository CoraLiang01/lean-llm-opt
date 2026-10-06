**Abstract Mathematical Model**

**Index Sets:**
- $I$: set of items (from file_0_view_0, column Item)
- $P$: set of cutting patterns (from file_1_view_0, column Pattern)

**Parameters:**
- $d_i$: demand for item $i \in I$ (from file_0_view_0, column Demand)
- $a_{pi}$: number of units of item $i$ produced by pattern $p$ (from file_1_view_0, columns $I$, rows indexed by Pattern)

**Decision Variables:**
- $y_p \in \mathbb{Z}_{\geq 0}$: number of times pattern $p \in P$ is used

**Objective:**
\[
\min \sum_{p \in P} y_p
\]

**Constraints:**
- **Demand satisfaction:**  
  For each item $i \in I$,
  \[
  \sum_{p \in P} a_{pi} y_p \geq d_i
  \]
- **Nonnegativity and integrality:**  
  For each pattern $p \in P$,
  \[
  y_p \in \mathbb{Z}_{\geq 0}
  \]

---

**Data Mapping**

- $I$: file_0_view_0, column Item
- $P$: file_1_view_0, column Pattern
- $d_i$: file_0_view_0, column Demand, key Item
- $a_{pi}$: file_1_view_0, columns Item, key Pattern
- $y_p$: decision variable for each $p \in P$