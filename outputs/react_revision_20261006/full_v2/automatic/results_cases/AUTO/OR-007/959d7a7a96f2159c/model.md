##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses, from supply_capacity.csv)
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores, from customer_demand.csv)

##### Parameters

- $d_j$: demand of store $j$ (from customer_demand.csv)
- $s_i$: supply capacity of warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

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

- $I$ (warehouses): region column in supply_capacity.csv (table_id: file_1_view_0)
- $J$ (stores): customer column in customer_demand.csv (table_id: file_0_view_0)
- $d_j$: demand column in customer_demand.csv (table_id: file_0_view_0)
- $s_i$: supply_capacity column in supply_capacity.csv (table_id: file_1_view_0)
- $c_{ij}$: entry in transportation_costs.csv (table_id: file_2_view_0), row Unnamed: 0 = $i$, column = $j$