#### Mathematical Model

Let:
- $I$ = set of distribution centers (suppliers) = {supplier1, supplier2, supplier3, supplier4, supplier5, supplier6, supplier7, supplier8}
- $J$ = set of customer groups (demands) = {demand1, demand2, demand3, demand4, demand5, demand6, demand7, demand8}
- $x_{ij} \geq 0$ = quantity shipped from supplier $i \in I$ to customer group $j \in J$ (continuous)
- $d_j$ = demand of customer group $j$ (from customer_demand.csv)
- $s_i$ = supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$ = transportation cost per unit from supplier $i$ to customer group $j$ (from transportation_costs.csv)

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

- $I$ (suppliers): All unique values in column supplier_id of file_1_view_0 (supply_capacity.csv)
- $J$ (customer groups): All unique values in column customer_id of file_0_view_0 (customer_demand.csv)
- $d_j$: Value in column demand for customer_id $j$ in file_0_view_0 (customer_demand.csv)
- $s_i$: Value in column supply_capacity for supplier_id $i$ in file_1_view_0 (supply_capacity.csv)
- $c_{ij}$: Value in column transportation_cost_to_${j} for supplier_id $i$ in file_2_view_0 (transportation_costs.csv), where ${j}$ matches customer_id in file_0_view_0

- Decision variables $x_{ij}$: For all $i \in I$, $j \in J$, continuous and nonnegative

All index sets, parameters, and constraints are defined directly from the current CSV data as described above. No data is omitted or aggregated.