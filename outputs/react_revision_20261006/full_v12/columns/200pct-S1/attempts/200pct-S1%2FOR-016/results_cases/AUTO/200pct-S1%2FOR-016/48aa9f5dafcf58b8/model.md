##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i \in I$ to customer group $j \in J$.

##### Sets

- $I$: set of distribution centers (supplier IDs from file_1_view_0 and file_2_view_0, e.g., $I = \{\text{S1}, \ldots, \text{S18}\}$)
- $J$: set of customer groups (customer IDs from file_0_view_0 and file_2_view_0, e.g., $J = \{\text{C1}, \ldots, \text{C18}\}$)

##### Parameters

- $d_j$: demand of customer group $j$ (from file_0_view_0, column demand_units)
- $s_i$: supply capacity of distribution center $i$ (from file_1_view_0, column supply_capacity_units)
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from file_2_view_0, column transportation_cost_to_$j$ for row $i$)

##### Objective

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

1. Demand satisfaction:
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. Supply capacity:
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. Non-negativity:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$: supplier_id from file_1_view_0 and file_2_view_0 (source order)
- $J$: customer_id from file_0_view_0 and file_2_view_0 (source order)
- $d_j$: file_0_view_0, column demand_units, indexed by customer_id
- $s_i$: file_1_view_0, column supply_capacity_units, indexed by supplier_id
- $c_{ij}$: file_2_view_0, row supplier_id $i$, column transportation_cost_to_$j$ (column_id_mapping in Observation)

All indices, parameters, and constraints are mapped directly to the current source data as described above.