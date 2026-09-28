### Abstract Mathematical Model

#### Index Sets
- $I$: Set of all products classified as ‘Aalop’.

#### Parameters
- $A_i$: Revenue per unit of product $i \in I$ (from column [Revenue]).
- $d_i$: Demand for product $i \in I$ (from column [Demand]).
- $I_i$: Initial inventory for product $i \in I$ (from column [Initial Inventory]).

#### Decision Variables
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

#### Objective
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints
1. Inventory constraint:
   $$
   x_i \leq I_i \quad \forall i \in I
   $$
2. Demand constraint:
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: RestaurantSalesreport.csv (table_id: file_0_view_0)
    - [Product Name]: Index set $I$ (products classified as ‘Aalop’)
    - [Revenue]: Parameter $A_i$
    - [Demand]: Parameter $d_i$
    - [Initial Inventory]: Parameter $I_i$