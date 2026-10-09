#### Mathematical Model

Let  
- $I$ = set of all products (indexed by $i$), as given by the "Product Name" column.

Parameters (for each $i \in I$):  
- $A_i$ = revenue per unit of product $i$ ("Revenue" column)  
- $d_i$ = total demand for product $i$ ("Demand" column)  
- $s_i$ = initial inventory of product $i$ ("Initial Inventory" column)

Decision variables:  
- $x_i$ = number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

Objective:  
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:  
- Demand and inventory bounds:
  $$
  0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in I
  $$
- Integrality:
  $$
  x_i \in \mathbb{Z}_+ \quad \forall i \in I
  $$

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv`
  - Index set $I$: all unique values in column `"Product Name"`
  - Parameter $A_i$: column `"Revenue"`
  - Parameter $d_i$: column `"Demand"`
  - Parameter $s_i$: column `"Initial Inventory"`