#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products classified as ‘Baby’ in the dataset.

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’)
- $d_i$: Demand for product $i$ (from column ‘Demand’)
- $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’)

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+, \; 0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**
1. **Demand fulfillment:**  
  $x_i \leq d_i \quad \forall i \in I$
2. **Inventory availability:**  
  $x_i \leq I_i \quad \forall i \in I$
3. **Non-negativity and integrality:**  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- **Table:** `file_0_view_0` (EuropeSalesRecords.csv)
- **Index Set $I$:** All records where ‘Product Name’ contains the substring ‘Baby’
- **Parameter $A_i$:** Column ‘Revenue’
- **Parameter $d_i$:** Column ‘Demand’
- **Parameter $I_i$:** Column ‘Initial Inventory’