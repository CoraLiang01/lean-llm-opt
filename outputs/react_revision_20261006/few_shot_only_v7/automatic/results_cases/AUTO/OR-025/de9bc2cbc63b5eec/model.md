#### Symbolic Model

**Index Sets:**  
Let $\mathcal{I}$ be the set of all products where `Product Name` contains 'TABLET' in table `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv`.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (`Revenue`)
- $d_i$: Deterministic demand for product $i$ (`Demand`)
- $I_i$: Initial inventory of product $i$ (`Initial Inventory`)

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i \in \mathbb{Z}_+, \quad$ number of units of product $i$ to fulfill

**Objective:**  
$\max \sum_{i \in \mathcal{I}} A_i x_i$

**Constraints:**  
For all $i \in \mathcal{I}$:
- $x_i \leq d_i$  (demand constraint)
- $x_i \leq I_i$  (inventory constraint)
- $x_i \geq 0$  (non-negativity and integrality)

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv`
    - Index set $\mathcal{I}$: rows where `Product Name` contains 'TABLET'
    - $A_i$: column `Revenue`
    - $d_i$: column `Demand`
    - $I_i$: column `Initial Inventory`