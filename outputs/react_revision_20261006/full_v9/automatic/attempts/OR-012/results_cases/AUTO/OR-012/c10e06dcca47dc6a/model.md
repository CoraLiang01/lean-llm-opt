#### Mathematical Optimization Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all products, where each $i \in \mathcal{I}$ corresponds to a unique "Product Name" in the dataset.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Deterministic demand for product $i$ over the sales horizon (from column "Demand")
- $I_i$: Initial inventory available for product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill for customer purchases  
  Domain: $x_i \in \mathbb{Z}_+, \quad 0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

**Constraints:**
1. **Demand fulfillment constraint:**  
   $$
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   $$
2. **Inventory constraint:**  
   $$
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   $$
3. **Non-negativity and integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   $$

---

#### Data Mapping

- **Index Set $\mathcal{I}$:** All unique values in column "Product Name" of table_id: file_0_view_0
- **Parameter $A_i$:** Column "Revenue" of table_id: file_0_view_0
- **Parameter $d_i$:** Column "Demand" of table_id: file_0_view_0
- **Parameter $I_i$:** Column "Initial Inventory" of table_id: file_0_view_0
- **Decision Variable $x_i$:** Defined for each $i \in \mathcal{I}$

All data is sourced from table_id: file_0_view_0, columns: "Product Name", "Revenue", "Demand", "Initial Inventory".