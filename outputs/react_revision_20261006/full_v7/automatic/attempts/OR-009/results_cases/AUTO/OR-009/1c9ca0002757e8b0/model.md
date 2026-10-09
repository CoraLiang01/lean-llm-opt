##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of beverages shipped from plant $i$ to retail outlet $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets)

##### Parameters

- $d_j$: demand at outlet $j$ (from customer_demand.csv)
- $s_i$: supply capacity at plant $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each retail outlet must receive at least its demand.
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity:** Each plant cannot ship more than its capacity.
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (plants): S1, S2, S3, S4 (from supply_capacity.csv, column "Unnamed: 0", table_id: file_1_view_0)
- $J$ (retail outlets): C1, C2, C3, C4 (from customer_demand.csv, column "customer", table_id: file_0_view_0)
- $d_j$ (demand): from customer_demand.csv, column "demand", table_id: file_0_view_0, indexed by "customer"
- $s_i$ (supply capacity): from supply_capacity.csv, column "supply_capacity", table_id: file_1_view_0, indexed by "Unnamed: 0"
- $c_{ij}$ (transportation cost): from transportation_costs.csv, columns "C1", "C2", "C3", "C4", table_id: file_2_view_0, rows indexed by "Unnamed: 0" (plant), columns indexed by customer

All indices, parameters, and coefficients are to be taken exactly as specified in the retrieved tables.