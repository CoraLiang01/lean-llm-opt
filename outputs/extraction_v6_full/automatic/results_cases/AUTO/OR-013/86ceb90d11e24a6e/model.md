#### Abstract Mathematical Optimization Model

**Index Set:**

- $I$ : Set of “4U” products, indexed by $i$.

**Parameters:**

- $A_i$ : Revenue per unit of product $i$.  
- $d_i$ : Total demand for product $i$ over the sales horizon.  
- $I_i$ : Initial inventory of product $i$.

**Decision Variables:**

- $x_i$ : Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$.

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraints:**  
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$

2. **Demand Constraints:**  
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

3. **Non-negativity and Integrality:**  
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

---

#### Data Mapping

- $I$ : All records in table_id file_0_view_0 where "Product Name" has prefix "4U".
- $A_i$ : file_0_view_0, column "Revenue", indexed by "Product Name".
- $d_i$ : file_0_view_0, column "Demand", indexed by "Product Name".
- $I_i$ : file_0_view_0, column "Initial Inventory", indexed by "Product Name".

(CSVQA_DATA bindings: see table_id file_0_view_0, columns "Product Name", "Revenue", "Demand", "Initial Inventory", filtered by "Product Name" prefix "4U")