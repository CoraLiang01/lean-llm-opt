##### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products whose "Product Name" contains '27in' in the source data.

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**  
$\max \sum_{i \in I} A_i x_i$

**Constraints:**
- $x_i \leq d_i \quad \forall i \in I$  (Demand cannot be exceeded)
- $x_i \leq I_i \quad \forall i \in I$  (Cannot sell more than initial inventory)
- $x_i \geq 0$ and integer $\forall i \in I$

---

##### Data Mapping

- **Index Set $I$:** All records in table_id `file_0_view_0` where column "Product Name" contains '27in'
- **Parameter $A_i$:** table_id `file_0_view_0`, column "Revenue"
- **Parameter $d_i$:** table_id `file_0_view_0`, column "Demand"
- **Parameter $I_i$:** table_id `file_0_view_0`, column "Initial Inventory"