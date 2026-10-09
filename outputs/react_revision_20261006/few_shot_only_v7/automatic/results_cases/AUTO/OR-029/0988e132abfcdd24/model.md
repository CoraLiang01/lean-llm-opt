#### Symbolic Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all products where the value in column ‘Product Name’ contains the substring “FAUX”.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: revenue per unit of product $i$ (from column ‘Revenue’)
- $d_i$: total demand for product $i$ (from column ‘Demand’)
- $I_i$: initial inventory for product $i$ (from column ‘Initial Inventory’)

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**  
$\max \sum_{i \in \mathcal{I}} A_i x_i$

**Constraints:**
1. Inventory bounds:  
$\quad x_i \leq I_i \quad \forall i \in \mathcal{I}$

2. Demand bounds:  
$\quad x_i \leq d_i \quad \forall i \in \mathcal{I}$

3. Nonnegativity and integrality:  
$\quad x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$

#### Data Mapping

- Index set $\mathcal{I}$: All records in table_id `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv` where column `Product Name` contains "FAUX".
- Parameter $A_i$: column `Revenue` in the same table.
- Parameter $d_i$: column `Demand` in the same table.
- Parameter $I_i$: column `Initial Inventory` in the same table.