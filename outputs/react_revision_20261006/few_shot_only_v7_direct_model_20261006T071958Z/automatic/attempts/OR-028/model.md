#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all product identifiers in the current table, i.e.,  
$I = \{$ all "Product Name" values in table_id file_0_view_0 $\}$.

**Parameters:**  
For each $i \in I$:
- $A_i$: revenue per unit of product $i$ ("Revenue" column)
- $d_i$: total demand for product $i$ ("Demand" column)
- $I_i$: initial inventory for product $i$ ("Initial Inventory" column)

**Decision Variables:**  
For each $i \in I$:
- $x_i$: number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers)

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in I} A_i x_i
$$

**Constraints:**
1. **Demand fulfillment:**  
   $x_i \leq d_i \quad \forall i \in I$
2. **Inventory limit:**  
   $x_i \leq I_i \quad \forall i \in I$
3. **Non-negativity and integrality:**  
   $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- **Index Set $I$:**  
  All values in column "Product Name" of table_id file_0_view_0

- **Parameter $A_i$:**  
  "Revenue" column of table_id file_0_view_0

- **Parameter $d_i$:**  
  "Demand" column of table_id file_0_view_0

- **Parameter $I_i$:**  
  "Initial Inventory" column of table_id file_0_view_0

- **Decision Variable $x_i$:**  
  Defined for each $i \in I$ as above

All data is mapped directly from the specified columns in table_id file_0_view_0. No additional constraints or synthetic scenario parameters are imposed beyond those described above.