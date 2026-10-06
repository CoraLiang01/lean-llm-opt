#### Index Sets

- $I$: Set of all products classified under ‘Baby’.

#### Parameters

- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i \in I$ over the sales horizon (from column ‘Demand’).
- $I_i$: Initial inventory of product $i \in I$ (from column ‘Initial Inventory’).

#### Decision Variables

- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

#### Objective

\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints

1. **Inventory Constraint:**  
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]

2. **Demand Constraint:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]

3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv`
    - Product Name: identifies set $I$ (select those classified as ‘Baby’)
    - Revenue: parameter $A_i$
    - Demand: parameter $d_i$
    - Initial Inventory: parameter $I_i$