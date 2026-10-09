#### Mathematical Model

Let:
- $I$ = set of distribution centers (suppliers): $\{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J$ = set of customer groups (demands): $\{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$
- $x_{ij} \geq 0$ = quantity shipped from supplier $i \in I$ to customer group $j \in J$ (continuous)
- $c_{ij}$ = unit transportation cost from supplier $i$ to customer group $j$
- $d_j$ = demand of customer group $j$
- $s_i$ = supply capacity of supplier $i$

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
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

#### Data Mapping

- $I$ (suppliers): All unique values in column "supplier_id" of table_id file_1_view_0 (supply_capacity.csv)
- $J$ (customer groups): All unique values in column "customer_id" of table_id file_0_view_0 (customer_demand.csv)
- $d_j$: For each $j \in J$, value in column "demand" of table_id file_0_view_0, row where "customer_id" = $j$
- $s_i$: For each $i \in I$, value in column "supply_capacity" of table_id file_1_view_0, row where "supplier_id" = $i$
- $c_{ij}$: For each $i \in I$, $j \in J$, value in table_id file_2_view_0 (transportation_costs.csv), row where "supplier_id" = $i$, column "transportation_cost_to_$j$" (column_id mapping: demand1, ..., demand8)

Index sets, parameters, and all coefficients are defined exactly as above from the current CSV data. No data is omitted or aggregated.