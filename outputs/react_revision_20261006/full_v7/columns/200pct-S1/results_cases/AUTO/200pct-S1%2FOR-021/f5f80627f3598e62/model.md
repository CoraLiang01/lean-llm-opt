##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of beverages shipped from production plant $i \in I$ to retail outlet $j \in J$ (continuous).

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets)

##### Parameters

- $d_j$: demand at outlet $j \in J$ (from customer_demand.csv)
- $s_i$: supply capacity at plant $i \in I$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. Demand satisfaction at each outlet:
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. Supply capacity at each plant:
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. Non-negativity:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ from supply_capacity.csv (file_1_view_0, column supplier_id)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ from customer_demand.csv (file_0_view_0, column customer_id)
- $d_j$ from customer_demand.csv (file_0_view_0, column demand, indexed by customer_id)
- $s_i$ from supply_capacity.csv (file_1_view_0, column supply_capacity, indexed by supplier_id)
- $c_{ij}$ from transportation_costs.csv (file_2_view_0, columns transportation_cost_to_C1, ..., transportation_cost_to_C4, rows indexed by supplier_id, columns mapped to customer_id as per relationships)

All indices, parameters, and coefficients are mapped directly from the retrieved CSV files as described above.