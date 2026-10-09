#### Mathematical Model

Let $I$ be the set of bread types (indexed by $i$), as given by the item_name column in products.csv.

**Parameters:**
- $v_i$: expected profit per unit of bread $i$ (from item_value, file_1_view_0)
- $a_i$: storage space required per unit of bread $i$ (from resource_requirement, file_1_view_0)
- $C$: total available storage capacity (from resource_capacity, file_0_view_0)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of bread $i$ to order each day

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} a_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

#### Data Mapping

- $I$: All item_name in file_1_view_0 (products.csv)
- $v_i$: item_value in file_1_view_0, mapped by item_name
- $a_i$: resource_requirement in file_1_view_0, mapped by item_name
- $C$: resource_capacity in file_0_view_0 (capacity.csv), row 0

Each $x_i$ is the integer number of units of bread $i$ to order. The total storage used cannot exceed $C$. The objective is to maximize total expected profit.