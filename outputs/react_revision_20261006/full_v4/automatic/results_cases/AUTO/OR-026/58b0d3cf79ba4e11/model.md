##### Mathematical Optimization Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all products classified under ‘Fashion’ in the dataset (as identified by the filter on "Product Name" in the source).

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue", table_id: file_0_view_0)
- $d_i$: Demand for product $i$ (from column "Demand", table_id: file_0_view_0)
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory", table_id: file_0_view_0)

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill (integer, $x_i \geq 0$)

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{I}} A_i x_i
$$

**Constraints:**
1. Inventory constraint for each product:
$$
x_i \leq I_i \quad \forall i \in \mathcal{I}
$$

2. Demand constraint for each product:
$$
x_i \leq d_i \quad \forall i \in \mathcal{I}
$$

3. Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \mathcal{I}
$$

---

##### Data Mapping

- **Index Set $\mathcal{I}$:** All records in table_id: file_0_view_0 ("SupermarketSales.csv") where "Product Name" starts with "Fashion".
- **Parameter $A_i$:** "Revenue" column, table_id: file_0_view_0.
- **Parameter $d_i$:** "Demand" column, table_id: file_0_view_0.
- **Parameter $I_i$:** "Initial Inventory" column, table_id: file_0_view_0.
- **Variable $x_i$:** Defined for each $i \in \mathcal{I}$ as above.