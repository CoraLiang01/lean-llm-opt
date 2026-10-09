##### Mathematical Model

Let:
- $I$ = set of distribution centers (suppliers), indexed by $i$, from all supplier_id in file_1_view_0.
- $J$ = set of customer groups, indexed by $j$, from all customer_id in file_0_view_0.
- $x_{ij} \geq 0$ = quantity shipped from distribution center $i$ to customer group $j$ (continuous).

Parameters:
- $d_j$ = demand of customer group $j$ (from file_0_view_0, column demand).
- $s_i$ = supply capacity of distribution center $i$ (from file_1_view_0, column supply_capacity).
- $c_{ij}$ = transportation cost per unit from $i$ to $j$ (from file_2_view_0, column transportation_cost_to_$j$ for supplier_id $i$).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
2. Supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (distribution centers): All supplier_id in file_1_view_0 (supply_capacity.csv), source order.
- $J$ (customer groups): All customer_id in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: file_0_view_0, column demand, for customer_id $j$.
- $s_i$: file_1_view_0, column supply_capacity, for supplier_id $i$.
- $c_{ij}$: file_2_view_0, row with supplier_id $i$, column transportation_cost_to_$j$ (e.g., transportation_cost_to_C1 for $j$ = C1).

- Decision variables $x_{ij}$: continuous, nonnegative, for all $i \in I$, $j \in J$.

All index sets, parameters, and constraints are defined directly from the current CSV data, preserving all identifiers and source order. No data is omitted or aggregated.