#### Abstract Mathematical Model

**Index Set:**

- $I$ : Set of all products with identifiers starting with ‘S700_’.

**Parameters:**

- $A_i$ : Revenue per unit for product $i \in I$ (from column ‘Revenue’).
- $d_i$ : Total demand for product $i \in I$ (from column ‘Demand’).
- $I_i$ : Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraint:** 
   $$
   x_i \leq I_i \quad \forall i \in I
   $$
2. **Demand Constraint:** 
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
3. **Non-negativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** `file_0_view_0` (from `SampleSalesData.csv`)
- **Columns Used:**
    - Product Identifier: `Product Name` (filtered by prefix ‘S700_’)
    - Revenue: `Revenue`
    - Demand: `Demand`
    - Initial Inventory: `Initial Inventory`
- **Filter Applied:** Only rows where `Product Name` starts with ‘S700_’ (as returned by CSVQA).