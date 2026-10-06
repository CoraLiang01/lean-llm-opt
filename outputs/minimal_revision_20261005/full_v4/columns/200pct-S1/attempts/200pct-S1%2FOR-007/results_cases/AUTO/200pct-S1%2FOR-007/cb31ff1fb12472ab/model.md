**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types (from `products.csv`, column `ProductName`).

**Parameters:**
- $p_i$: Profit per unit of vehicle $i$ (from `products.csv`, column `Value`).
- $w_i$: Inventory space required per unit of vehicle $i$ (from `products.csv`, column `Weight`).
- $C$: Total inventory capacity (from `capacity.csv`, column `Capacity`).

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i$ to order per day.

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

- $I$: All records in `file_1_view_0` (`products.csv`), column `ProductName`.
- $p_i$: `file_1_view_0`, column `Value`, keyed by `ProductName`.
- $w_i$: `file_1_view_0`, column `Weight`, keyed by `ProductName`.
- $C$: `file_0_view_0`, column `Capacity`.

---

**Summary:**  
Maximize total profit from ordering vehicles, subject to the overall inventory capacity. Decision variables are the integer number of each vehicle type to order per day. All parameters and index sets are mapped directly from the supplied data.