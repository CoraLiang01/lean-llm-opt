## Abstract Mathematical Model

**Sets:**
- $S$: set of sections (indexed by $s$), from file_0_view_0[SectionID]
- $P$: set of products (indexed by $p$), from file_1_view_0[ProductName]

**Parameters:**
- $c_s$: display space capacity of section $s$, from file_0_view_0[Capacity]
- $v_p$: price (revenue per unit) of product $p$, from file_1_view_0[Value]
- $w_p$: shelf space requirement per unit of product $p$, from file_1_view_0[Weight]

**Decision Variables:**
- $x_{sp}$: number of units of product $p$ to stock in section $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
- Section capacity constraints (for all $s \in S$):
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq c_s
\]
- Integer and nonnegativity constraints (for all $s \in S$, $p \in P$):
\[
x_{sp} \in \mathbb{Z}_{\geq 0}
\]

---

## Data Mapping

- $S$ (sections): file_0_view_0[SectionID]
- $c_s$: file_0_view_0[Capacity], keyed by SectionID
- $P$ (products): file_1_view_0[ProductName]
- $v_p$: file_1_view_0[Value], keyed by ProductName
- $w_p$: file_1_view_0[Weight], keyed by ProductName

All data is used as returned, preserving file and row order.