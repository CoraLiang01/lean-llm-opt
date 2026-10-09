#### Symbolic Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all products where Product_Name contains 'Books'.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column 'Revenue')
- $d_i$: Demand for product $i$ (from column 'Demand')
- $I_i$: Initial inventory for product $i$ (from column 'Initial Inventory')

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i \in \mathbb{Z}_+, \quad 0 \leq x_i \leq \min\{d_i, I_i\}$: Number of units of product $i$ to fulfill

**Objective:**  
$\max \sum_{i \in \mathcal{I}} A_i x_i$

**Constraints:**  
For all $i \in \mathcal{I}$:
- $x_i \leq d_i$  (Demand constraint)
- $x_i \leq I_i$  (Inventory constraint)
- $x_i \geq 0$  (Non-negativity and integrality)

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv`
    - Index set $\mathcal{I}$: rows where `Product_Name` contains 'Books'
    - Parameter $A_i$: column `Revenue`
    - Parameter $d_i$: column `Demand`
    - Parameter $I_i$: column `Initial Inventory`