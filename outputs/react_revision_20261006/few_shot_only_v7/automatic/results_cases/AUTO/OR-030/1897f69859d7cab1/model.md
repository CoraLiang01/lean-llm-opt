#### Sets
- $I$: Index set of all car models with `Product Name` classified as 'FDK57'.

#### Parameters
- $A_i$: Revenue per unit of car model $i \in I$ (`Revenue` column).
- $d_i$: Deterministic demand for car model $i \in I$ (`Demand` column).
- $I_i$: Initial inventory for car model $i \in I$ (`Initial Inventory` column).

#### Decision Variables
- $x_i$: Number of units of car model $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

#### Objective
$$
\max \sum_{i \in I} A_i x_i
$$

#### Constraints
1. **Inventory constraint:** 
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. **Demand constraint:** 
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. **Non-negativity and integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

**Data Mapping**

- Source Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv`
    - Index set $I$: All rows where `Product Name` = 'FDK57'
    - $A_i$: `Revenue` column
    - $d_i$: `Demand` column
    - $I_i$: `Initial Inventory` column