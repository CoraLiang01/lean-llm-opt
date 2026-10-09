#### Symbolic Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all products in the dataset with Product_Reference starting with 'ELE-S'.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column 'Revenue')
- $d_i$: Total demand for product $i$ (from column 'Demand')
- $I_i$: Initial inventory of product $i$ (from column 'Initial Inventory')

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i \in \mathbb{Z}_+, \quad$ number of units of product $i$ to fulfill

**Objective:**  
$\max \quad \sum_{i \in \mathcal{I}} A_i \cdot x_i$

**Constraints:**
1. Inventory bounds:  
$\quad x_i \leq I_i \quad \forall i \in \mathcal{I}$

2. Demand bounds:  
$\quad x_i \leq d_i \quad \forall i \in \mathcal{I}$

3. Nonnegativity and integrality:  
$\quad x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$

---

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv`
    - Index set $\mathcal{I}$: rows where `Product_Reference` starts with 'ELE-S'
    - Parameter $A_i$: column `Revenue`
    - Parameter $d_i$: column `Demand`
    - Parameter $I_i$: column `Initial Inventory`