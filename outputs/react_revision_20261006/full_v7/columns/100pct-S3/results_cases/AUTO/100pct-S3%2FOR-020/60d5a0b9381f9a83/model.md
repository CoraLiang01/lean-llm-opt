##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses)
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores)

##### Parameters

- $d_j$: demand at store $j \in J$
- $s_i$: supply capacity at warehouse $i \in I$
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each store's demand must be met:
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity:** Each warehouse cannot ship more than its capacity:
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (warehouses): supplier_id from supply_capacity.csv and transportation_costs.csv
- $J$ (stores): customer_id from customer_demand.csv and transportation_costs.csv
- $d_j$: demand_units from customer_demand.csv, for each $j \in J$
- $s_i$: supply_capacity_units from supply_capacity.csv, for each $i \in I$
- $c_{ij}$: transportation_cost_to_Dk from transportation_costs.csv, where $i$ = supplier_id, $j$ = Dk

All indices, parameters, and coefficients are mapped directly from the provided CSV files as described above.