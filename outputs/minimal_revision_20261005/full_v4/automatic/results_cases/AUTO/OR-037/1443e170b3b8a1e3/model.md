**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types (from `file_1_view_0`, column `ProductName`)

**Parameters:**
- $p_i$: Profit per unit of vehicle $i$ (from `file_1_view_0`, column `Value`)
- $w_i$: Inventory space required per unit of vehicle $i$ (from `file_1_view_0`, column `Weight`)
- $C$: Total inventory capacity (from `file_0_view_0`, column `Capacity`)

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day; $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Constraints:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

**Data Mapping**

- $I$: All records in `file_1_view_0`, column `ProductName`
- $p_i$: `file_1_view_0`, column `Value`, keyed by `ProductName`
- $w_i$: `file_1_view_0`, column `Weight`, keyed by `ProductName`
- $C$: `file_0_view_0`, column `Capacity` (single value)