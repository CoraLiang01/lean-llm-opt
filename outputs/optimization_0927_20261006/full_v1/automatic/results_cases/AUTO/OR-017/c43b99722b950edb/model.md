#### Abstract Mathematical Optimization Model

**Index Set:**

- $I$ : Set of all products classified under ‘ZZ’ (indexed by $i$).

**Parameters:**

- $A_i$ : Revenue per unit of product $i$ (from column ‘Revenue’ in table_id: file_0_view_0).
- $d_i$ : Total demand for product $i$ (from column ‘Demand’ in table_id: file_0_view_0).
- $I_i$ : Initial inventory of product $i$ (from column ‘Initial Inventory’ in table_id: file_0_view_0).

**Decision Variables:**

- $x_i$ : Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$, for all $i \in I$.

**Objective Function:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraint:** 
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. **Demand Constraint:** 
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. **Nonnegativity and Integrality:** 
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** file_0_view_0 (from RetailStoreSalesTransactions(ScannerData).csv)
- **Index Set:** $I$ is defined by all rows where column ‘SKU’ has prefix ‘ZZ’.
- **Parameters:**
    - $A_i$ : column ‘Revenue’
    - $d_i$ : column ‘Demand’
    - $I_i$ : column ‘Initial Inventory’
- **Decision Variables:** $x_i$ corresponds to each $i \in I$ (each ‘ZZ’ SKU).