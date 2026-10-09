Mathematical Model

Sets:
Let $I$ be the set of distribution centers (suppliers), indexed by $i$:
$I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$

Let $J$ be the set of customer groups, indexed by $j$:
$J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

Parameters:
$d_j$ = demand of customer group $j$ (from file_0_view_0, column "demand", key "customer_id")
$s_i$ = supply capacity of distribution center $i$ (from file_1_view_0, column "supply_capacity", key "supplier_id")
$c_{ij}$ = transportation cost per unit from distribution center $i$ to customer group $j$ (from file_2_view_0, column "transportation_cost_to_$j$", key "supplier_id")

Decision Variables:
$x_{ij} \geq 0$ = quantity shipped from distribution center $i$ to customer group $j$ (continuous)

Objective:
Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:

1. Demand satisfaction for each customer group:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Supply capacity for each distribution center:
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

Data Mapping

- $I$ (distribution centers): All "supplier_id" in file_1_view_0 (supply_capacity.csv), source order.
- $J$ (customer groups): All "customer_id" in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: file_0_view_0, column "demand", key "customer_id".
- $s_i$: file_1_view_0, column "supply_capacity", key "supplier_id".
- $c_{ij}$: file_2_view_0, row "supplier_id" = $i$, column "transportation_cost_to_$j$" (column_id_mapping: e.g., "transportation_cost_to_C1" $\rightarrow$ "C1").

All sets, parameters, and indices are defined by the current CSV data, preserving source order and identifiers. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.