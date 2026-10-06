#### Index Sets

- $I$: Set of all products classified as ‘Books’ (from column Product_Name).

#### Parameters

- $a_i$: Revenue per unit of product $i \in I$ (from column Revenue).
- $d_i$: Demand for product $i \in I$ (from column Demand).
- $s_i$: Initial inventory for product $i \in I$ (from column Initial Inventory).

#### Decision Variables

- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \geq 0$, integer.

#### Objective

$$
\max \sum_{i \in I} a_i x_i
$$

#### Constraints

1. **Inventory Constraint:**  
   $x_i \leq s_i \quad \forall i \in I$

2. **Demand Constraint:**  
   $x_i \leq d_i \quad \forall i \in I$

3. **Nonnegativity and Integrality:**  
   $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv`
    - Index set $I$: rows where `Product_Name` indicates ‘Books’
    - $a_i$: `Revenue`
    - $d_i$: `Demand`
    - $s_i$: `Initial Inventory`