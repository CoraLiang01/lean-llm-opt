#### Symbolic Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all products classified under ‘Baby’ in the dataset.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’)
- $d_i$: Deterministic demand for product $i$ (from column ‘Demand’)
- $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’)

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**  
$\displaystyle \max \sum_{i \in \mathcal{I}} A_i x_i$

**Constraints:**
1. **Inventory Bound:**  
   $x_i \leq I_i \quad \forall i \in \mathcal{I}$
2. **Demand Bound:**  
   $x_i \leq d_i \quad \forall i \in \mathcal{I}$
3. **Nonnegativity and Integrality:**  
   $x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$

---

#### Data Mapping

- **Index Set $\mathcal{I}$:** All records in table_id `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv` where `Product Name` contains the substring ‘Baby’.
- **Parameter $A_i$:** Column `Revenue` in the same table.
- **Parameter $d_i$:** Column `Demand` in the same table.
- **Parameter $I_i$:** Column `Initial Inventory` in the same table.