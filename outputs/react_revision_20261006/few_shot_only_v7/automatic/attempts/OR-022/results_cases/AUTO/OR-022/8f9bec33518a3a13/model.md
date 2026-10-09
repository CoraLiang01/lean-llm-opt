#### Symbolic Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all products where "Product Name" contains '27in'.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i \in \mathbb{Z}_+, \quad$ number of units of product $i$ to fulfill

**Objective:**  
$\max \quad \sum_{i \in \mathcal{I}} A_i \cdot x_i$

**Constraints:**
1. Inventory and Demand Bounds:  
$\quad x_i \leq I_i, \quad \forall i \in \mathcal{I}$  
$\quad x_i \leq d_i, \quad \forall i \in \mathcal{I}$

2. Nonnegativity and Integrality:  
$\quad x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$

---

#### Data Mapping

- **Source Table:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv
- **Index Set:** Products where "Product Name" contains '27in'
- **Parameter $A_i$:** Column "Revenue"
- **Parameter $d_i$:** Column "Demand"
- **Parameter $I_i$:** Column "Initial Inventory"