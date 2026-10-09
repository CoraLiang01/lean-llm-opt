#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products with identifiers beginning with ‘S700_’ in column “Product Name” of table `file_0_view_0`.

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column “Revenue”)
- $d_i$: Demand for product $i$ (from column “Demand”)
- $I_i$: Initial inventory of product $i$ (from column “Initial Inventory”)

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers)

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**
1. **Inventory Constraint:**  
  For all $i \in I$,
$$
x_i \leq I_i
$$

2. **Demand Constraint:**  
  For all $i \in I$,
$$
x_i \leq d_i
$$

3. **Non-negativity and Integrality:**  
  For all $i \in I$,
$$
x_i \in \mathbb{Z}_+, \quad x_i \geq 0
$$

---

#### Data Mapping

- **Table:** `file_0_view_0` (from `SampleSalesData.csv`)
- **Index Set $I$:** All rows where “Product Name” starts with ‘S700_’
- **Parameter $A_i$:** Column “Revenue”
- **Parameter $d_i$:** Column “Demand”
- **Parameter $I_i$:** Column “Initial Inventory”