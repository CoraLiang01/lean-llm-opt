## Integer Cutting-Stock Pattern-Selection Model

**Sets:**
- $I$: set of item types (from file_0_view_0, column "Item")
- $P$: set of cutting patterns (from file_1_view_0, column "Pattern")

**Parameters:**
- $d_i$: demand for item $i \in I$ (from file_0_view_0, column "Demand")
- $a_{pi}$: number of units of item $i$ produced by pattern $p$ (from file_1_view_0, columns "A", "B", "C", "D")

**Decision Variables:**
- $y_p \in \mathbb{Z}_+, \quad \forall p \in P$: number of standard rolls cut using pattern $p$

**Objective:**
\[
\min \sum_{p \in P} y_p
\]

**Constraints:**
\[
\sum_{p \in P} a_{pi} y_p \geq d_i, \quad \forall i \in I
\]
\[
y_p \geq 0 \text{ and integer}, \quad \forall p \in P
\]

---

### Data Mapping

- $I$ = all "Item" in file_0_view_0 ("A", "B", "C", "D")
- $P$ = all "Pattern" in file_1_view_0 ("P1", ..., "P9")
- $d_i$ = "Demand" in file_0_view_0, for each $i$
- $a_{pi}$ = entry in file_1_view_0, row "Pattern" $p$, column $i$
- $y_p$ = integer variable for each $p$ in $P$