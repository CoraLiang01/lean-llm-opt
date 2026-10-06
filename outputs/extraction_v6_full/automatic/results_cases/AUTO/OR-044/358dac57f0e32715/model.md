## Abstract Mathematical Model

**Sets:**
- $S$: set of sections, indexed by $s$ (SectionID from file_0_view_0)
- $P$: set of products, indexed by $p$ (ProductName from file_1_view_0)

**Parameters:**
- $c_s$: display space capacity of section $s$ (capacity, file_0_view_0, SectionID)
- $v_p$: price of product $p$ (price, file_1_view_0, ProductName)
- $w_p$: shelf space requirement of product $p$ (shelf_space, file_1_view_0, ProductName)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ to stock in section $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Constraints:**
- Section capacity constraints:
\[
\sum_{p \in P} w_p \, x_{sp} \leq c_s \qquad \forall s \in S
\]
- Integrality:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

---

## Data Mapping

- $c_s$: file_0_view_0, column SectionID (business key), value column Capacity
- $v_p$: file_1_view_0, column ProductName (business key), value column Value
- $w_p$: file_1_view_0, column ProductName (business key), value column Weight

Each $x_{sp}$ is indexed by (SectionID, ProductName) as per the original files. No data is omitted or synthesized.