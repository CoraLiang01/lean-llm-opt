#### Symbolic Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all products with "Aalop" in the "Product Name" column.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from "Revenue")
- $d_i$: Demand for product $i$ (from "Demand")
- $I_i$: Initial inventory of product $i$ (from "Initial Inventory")

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**  
$\max \sum_{i \in \mathcal{I}} A_i x_i$

**Constraints:**
1. Inventory constraint:  
$\quad x_i \leq I_i \quad \forall i \in \mathcal{I}$

2. Demand constraint:  
$\quad x_i \leq d_i \quad \forall i \in \mathcal{I}$

3. Nonnegativity and integrality:  
$\quad x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$

---

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv`
    - Index set $\mathcal{I}$: Filtered from "Product Name" column for entries containing 'Aalop'
    - Parameter $A_i$: "Revenue" column
    - Parameter $d_i$: "Demand" column
    - Parameter $I_i$: "Initial Inventory" column