##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups (customers), as defined by the source data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer group $j$ (from customer_demand.csv).
- $s_i$: supply capacity of distribution center $i$ (from supply_capacity.csv).
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from transportation_costs.csv).

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

- $I$ (distribution centers): All supplier_id in supply_capacity.csv (file_1_view_0), source order: S1, S2, ..., S12.
- $J$ (customer groups): All customer_id in customer_demand.csv (file_0_view_0), source order: C1, C2, ..., C12.
- $d_j$: demand for customer $j$ from column "demand" in customer_demand.csv (file_0_view_0), mapped by customer_id.
- $s_i$: supply capacity for supplier $i$ from column "supply_capacity" in supply_capacity.csv (file_1_view_0), mapped by supplier_id.
- $c_{ij}$: transportation cost from supplier $i$ to customer $j$ from column "transportation_cost_to_{j}" in transportation_costs.csv (file_2_view_0), mapped by supplier_id and customer_id.

Index sets, parameters, and cost matrix are defined by the exact identifiers and source order in the current CSV files. No data is omitted or aggregated. All constraints and variable domains are as specified in the user query and source data.