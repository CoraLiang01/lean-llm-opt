##### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products in the current data whose "Product Name" contains the substring "Baby".

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**  
$\max \sum_{i \in I} A_i \cdot x_i$

**Constraints:**
1. Inventory and demand bounds:
   - $x_i \leq I_i \quad \forall i \in I$
   - $x_i \leq d_i \quad \forall i \in I$
2. Nonnegativity and integrality:
   - $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

##### Data Mapping

- **Index set $I$:** All records in table_id `file_0_view_0` where column `"Product Name"` contains the substring `"Baby"`.
- **Parameter $A_i$:** From column `"Revenue"` in table_id `file_0_view_0`.
- **Parameter $d_i$:** From column `"Demand"` in table_id `file_0_view_0`.
- **Parameter $I_i$:** From column `"Initial Inventory"` in table_id `file_0_view_0`.