##### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants) and $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets).

Let $x_{ij} \geq 0$ be the quantity shipped from plant $i \in I$ to outlet $j \in J$ (continuous).

Parameters:
- $d_j$: demand at outlet $j$ (from customer_demand.csv)
- $s_i$: supply capacity at plant $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction (for each $j \in J$):
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]
2. Supply capacity (for each $i \in I$):
   \[
   \sum_{j \in J} x_{ij} \leq s_i
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (from supply_capacity.csv and transportation_costs.csv, column: supplier_id)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (from customer_demand.csv and transportation_costs.csv, columns: customer_id, transportation_cost_to_C*)
- $d_j$ from customer_demand.csv, column: demand, indexed by customer_id
- $s_i$ from supply_capacity.csv, column: supply_capacity, indexed by supplier_id
- $c_{ij}$ from transportation_costs.csv, columns: transportation_cost_to_C*, indexed by supplier_id and customer_id

All indices, parameters, and coefficients are mapped directly from the current CSV files as described above.