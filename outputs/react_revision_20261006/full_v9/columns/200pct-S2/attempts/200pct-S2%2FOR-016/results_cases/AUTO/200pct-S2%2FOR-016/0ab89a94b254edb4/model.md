#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the identifiers in the source data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer group $j$ (from "customer_demand.csv", column "demand_units", indexed by "customer_id").
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv", column "supply_capacity_units", indexed by "supplier_id").
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from "transportation_costs.csv", column "transportation_cost_to_Ck" for customer $Ck$, row "supplier_id" $i$).

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

- $I$: All "supplier_id" in "supply_capacity.csv" (table_id: file_1_view_0, column: supplier_id)
- $J$: All "customer_id" in "customer_demand.csv" (table_id: file_0_view_0, column: customer_id)
- $d_j$: "demand_units" for customer $j$ in "customer_demand.csv" (table_id: file_0_view_0, columns: customer_id, demand_units)
- $s_i$: "supply_capacity_units" for supplier $i$ in "supply_capacity.csv" (table_id: file_1_view_0, columns: supplier_id, supply_capacity_units)
- $c_{ij}$: "transportation_cost_to_Ck" for supplier $i$ and customer $j = Ck$ in "transportation_costs.csv" (table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_Ck)

Index sets, parameters, and cost matrix are defined exactly by the current source data, preserving all identifiers and coefficients. No data is omitted or aggregated.