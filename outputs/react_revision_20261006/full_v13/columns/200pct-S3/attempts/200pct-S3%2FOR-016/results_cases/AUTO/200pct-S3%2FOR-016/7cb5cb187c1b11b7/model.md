#### Mathematical Model

Let:
- $I$ = set of distribution centers (suppliers), indexed by $i$, from all supplier_id in file_1_view_0.
- $J$ = set of customer groups, indexed by $j$, from all customer_id in file_0_view_0.
- $x_{ij} \geq 0$ = quantity of goods shipped from distribution center $i$ to customer group $j$ (continuous).

Parameters:
- $d_j$ = demand of customer group $j$ (demand_units from file_0_view_0).
- $s_i$ = supply capacity of distribution center $i$ (supply_capacity_units from file_1_view_0).
- $c_{ij}$ = transportation cost per unit from $i$ to $j$ (transportation_cost_to_Ck columns in file_2_view_0, with $i$ = supplier_id, $j$ = customer_id).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each customer group:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
2. Supply capacity for each distribution center:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$ (distribution centers): All supplier_id in table_id file_1_view_0 (supply_capacity.csv), source order.
- $J$ (customer groups): All customer_id in table_id file_0_view_0 (customer_demand.csv), source order.
- $d_j$: demand_units for customer_id $j$ in file_0_view_0.
- $s_i$: supply_capacity_units for supplier_id $i$ in file_1_view_0.
- $c_{ij}$: transportation_cost_to_Ck for supplier_id $i$ and customer_id $j$ in file_2_view_0, with column mapping as per relationships in the Observation.
- $x_{ij}$: continuous, nonnegative, indexed by all $(i,j) \in I \times J$.

All index sets, parameters, and coefficients are defined exactly as returned in the current Observation, preserving source order and identifiers. No data is omitted or aggregated.