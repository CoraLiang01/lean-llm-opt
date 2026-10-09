#### Symbolic Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all products classified as ‘Baby’ in the dataset.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’)
- $d_i$: Total demand for product $i$ over the sales horizon (from column ‘Demand’)
- $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’)

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**  
$\max \sum_{i \in \mathcal{I}} A_i x_i$

**Constraints:**
1. Inventory constraint: $x_i \leq I_i \quad \forall i \in \mathcal{I}$
2. Demand constraint:  $x_i \leq d_i \quad \forall i \in \mathcal{I}$
3. Nonnegativity and integrality: $x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv`
    - Index set $\mathcal{I}$: All rows where ‘Product Name’ is classified as ‘Baby’
    - Parameter $A_i$: Column ‘Revenue’
    - Parameter $d_i$: Column ‘Demand’
    - Parameter $I_i$: Column ‘Initial Inventory’