#### Integer Cutting-Stock Pattern-Selection Model

**Index Sets:**
- $I$: set of item types (from `file_0_view_0.Item`)
- $P$: set of cutting patterns (from `file_1_view_0.Pattern`)

**Parameters:**
- $d_i$: demand for item $i \in I$ (from `file_0_view_0.Demand`)
- $a_{pi}$: number of units of item $i$ produced by one standard roll cut with pattern $p \in P$ (from `file_1_view_0`, columns $I$, rows $P$)

**Decision Variables:**
- $y_p \in \mathbb{Z}_{\geq 0}$: number of standard rolls cut using pattern $p \in P$

**Objective:**
\[
\min \sum_{p \in P} y_p
\]

**Constraints:**
\[
\sum_{p \in P} a_{pi} y_p \geq d_i \qquad \forall i \in I
\]
\[
y_p \in \mathbb{Z}_{\geq 0} \qquad \forall p \in P
\]

---

**Data Mapping:**

- $I$: All `Item` values from `file_0_view_0` (`item_demand.csv`)
- $P$: All `Pattern` values from `file_1_view_0` (`cutting_patterns.csv`)
- $d_i$: `Demand` column in `file_0_view_0`, keyed by `Item`
- $a_{pi}$: Entry in `file_1_view_0` at row `Pattern` $p$, column $i$ (`A`, `B`, `C`, `D`)
- $y_p$: Decision variable for each $p \in P$ (pattern in `file_1_view_0.Pattern`)

All index sets, parameters, and mappings are defined directly from the returned CSV data.