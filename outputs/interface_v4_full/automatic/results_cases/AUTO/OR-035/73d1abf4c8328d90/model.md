#### Abstract Mathematical Model

**Sets:**
- $I$: Set of bread types, indexed by $i$. (From products.csv, column ProductName)

**Parameters:**
- $v_i$: Expected profit per unit of bread type $i$. (products.csv, column Value)
- $w_i$: Storage weight per unit of bread type $i$. (products.csv, column Weight)
- $C$: Total storage capacity. (capacity.csv, column Capacity)

**Decision Variables:**
- $x_i$: Number of units of bread type $i$ to order each day. ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

#### Data Mapping

- $I$: All records in file_1_view_0 (products.csv), column ProductName.
- $v_i$: file_1_view_0 (products.csv), column Value, keyed by ProductName.
- $w_i$: file_1_view_0 (products.csv), column Weight, keyed by ProductName.
- $C$: file_0_view_0 (capacity.csv), column Capacity.

Each $x_i$ is the integer number of units to order for bread type $i$ (ProductName). The total storage used, $\sum_{i} w_i x_i$, must not exceed the single storage capacity $C$ from capacity.csv. The objective is to maximize total expected profit.