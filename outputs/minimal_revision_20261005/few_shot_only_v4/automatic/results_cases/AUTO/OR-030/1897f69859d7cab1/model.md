#### Abstract Mathematical Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all car models classified under ‘FDK57’ in the dataset.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of car model $i$ (from column ‘Revenue’)
- $d_i$: Total deterministic demand for car model $i$ (from column ‘Demand’)
- $I_i$: Initial inventory for car model $i$ (from column ‘Initial Inventory’)

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of car model $i$ to fulfill (integer, $x_i \geq 0$)

**Objective:**  
$$
\max \sum_{i \in \mathcal{I}} A_i x_i
$$

**Constraints:**
1. **Inventory Constraint:**  
   $$
   x_i \leq I_i \qquad \forall i \in \mathcal{I}
   $$
2. **Demand Constraint:**  
   $$
   x_i \leq d_i \qquad \forall i \in \mathcal{I}
   $$
3. **Non-negativity and Integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \qquad \forall i \in \mathcal{I}
   $$

---

#### Data Mapping

- **Source Table:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv`
- **Index Set:** All rows where `Product Name` is classified under ‘FDK57’
- **Parameters:**
  - $A_i$: Column `Revenue`
  - $d_i$: Column `Demand`
  - $I_i$: Column `Initial Inventory`