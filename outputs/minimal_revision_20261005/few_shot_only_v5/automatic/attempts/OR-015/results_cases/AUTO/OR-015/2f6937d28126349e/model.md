#### Sets
- $I$ : set of all products classified under ‘Aalop’.

#### Parameters
- $A_i$ : revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$ : total demand for product $i \in I$ over the sales horizon (from column ‘Demand’).
- $I_i$ : initial inventory of product $i \in I$ (from column ‘Initial Inventory’).

#### Decision Variables
- $x_i$ : number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

#### Objective
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraint** (cannot sell more than initial inventory):
   $$
   x_i \leq I_i \quad \forall i \in I
   $$

2. **Demand Constraint** (cannot fulfill more than demand):
   $$
   x_i \leq d_i \quad \forall i \in I
   $$

3. **Non-negativity and Integrality**:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv`
    - Product Name: index set $I$
    - Revenue: parameter $A_i$
    - Demand: parameter $d_i$
    - Initial Inventory: parameter $I_i$