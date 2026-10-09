#### Mathematical Model

Let:
- $I$ = set of distribution centers (suppliers), indexed by $i$, from all supplier_id in file_1_view_0.
- $J$ = set of customer groups, indexed by $j$, from all customer_id in file_0_view_0.
- $x_{ij} \geq 0$ = quantity shipped from distribution center $i$ to customer group $j$ (continuous).

Parameters:
- $d_j$ = demand of customer group $j$ (demand_units from file_0_view_0).
- $s_i$ = supply capacity of distribution center $i$ (supply_capacity_units from file_1_view_0).
- $c_{ij}$ = transportation cost per unit from $i$ to $j$ (transportation_cost_to_Ck columns in file_2_view_0, where $k$ matches $j$).

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

- $I$ (distribution centers): All supplier_id in table_id file_1_view_0 (file_index 1, column "supplier_id").
- $J$ (customer groups): All customer_id in table_id file_0_view_0 (file_index 0, column "customer_id").
- $d_j$: For each $j \in J$, demand_units from file_0_view_0 (file_index 0, column "demand_units", key "customer_id").
- $s_i$: For each $i \in I$, supply_capacity_units from file_1_view_0 (file_index 1, column "supply_capacity_units", key "supplier_id").
- $c_{ij}$: For each $i \in I$, $j \in J$, value in file_2_view_0 (file_index 2) at row with supplier_id $i$ and column "transportation_cost_to_$j$" (where $j$ is the customer_id).

- Decision variables $x_{ij}$: continuous, nonnegative, for all $i \in I$, $j \in J$.

All index sets, parameters, and coefficients are defined exactly as above from the current CSV data. No data is omitted or aggregated.