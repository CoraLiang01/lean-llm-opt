#### Symbolic Mathematical Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all products where the value in column ‘Sub Category’ contains the substring “Organ”.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’)
- $d_i$: Total demand for product $i$ (from column ‘Demand’)
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’)

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

**Data Mapping:**  
- Table: SupermartGrocerySales-RetailAnalyticsDataset.csv
- Index set $\mathcal{I}$: All rows where ‘Sub Category’ contains “Organ”
- $A_i$: column ‘Revenue’
- $d_i$: column ‘Demand’
- $I_i$: column ‘Initial Inventory’