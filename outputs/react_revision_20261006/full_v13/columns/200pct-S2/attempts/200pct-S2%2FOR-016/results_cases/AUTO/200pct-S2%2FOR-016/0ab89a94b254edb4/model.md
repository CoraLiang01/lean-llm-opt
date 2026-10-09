#### Abstract Symbolic Formulation

Let:
- $I$ = set of distribution centers (suppliers), indexed by $i$, from all unique values in column "supplier_id" of table_id "file_1_view_0".
- $J$ = set of customer groups, indexed by $j$, from all unique values in column "customer_id" of table_id "file_0_view_0$.
- $d_j$ = demand of customer $j$, from column "demand_units" in table_id "file_0_view_0".
- $s_i$ = supply capacity of supplier $i$, from column "supply_capacity_units" in table_id "file_1_view_0".
- $c_{ij}$ = transportation cost per unit from supplier $i$ to customer $j$, from column "transportation_cost_to_{j}" in table_id "file_2_view_0", row "supplier_id" = $i$.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous).

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. Demand satisfaction:
$$
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
$$

2. Supply capacity:
$$
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
$$

#### Data Mapping

- $I$ (suppliers): all "supplier_id" in table_id "file_1_view_0" (supply_capacity.csv), source order.
- $J$ (customers): all "customer_id" in table_id "file_0_view_0" (customer_demand.csv), source order.
- $d_j$: "demand_units" for customer $j$ in table_id "file_0_view_0", source order.
- $s_i$: "supply_capacity_units" for supplier $i$ in table_id "file_1_view_0", source order.
- $c_{ij}$: "transportation_cost_to_{j}" for supplier $i$ in table_id "file_2_view_0" (transportation_costs.csv), where $j$ matches the customer_id in $J$ and $i$ matches the supplier_id in $I$; source order for both axes.
- $x_{ij}$: continuous, nonnegative, for all $i \in I$, $j \in J$.

All index sets, parameters, and coefficients are mapped directly from the current CSV files as described above, preserving source order and identifiers. No data is omitted or aggregated.