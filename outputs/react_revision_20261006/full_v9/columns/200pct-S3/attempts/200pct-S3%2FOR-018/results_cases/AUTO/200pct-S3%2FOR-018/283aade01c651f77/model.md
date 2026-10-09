##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the current data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer group $j \in J$.
- $s_i$: supply capacity of distribution center $i \in I$.
- $c_{ij}$: transportation cost per unit from $i$ to $j$.

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

- $I$ (distribution centers): All supplier_id in supply_capacity.csv (file_1_view_0), source order.
- $J$ (customer groups): All customer_id in customer_demand.csv (file_0_view_0), source order.
- $d_j$: demand column in customer_demand.csv (file_0_view_0), indexed by customer_id.
- $s_i$: supply_capacity column in supply_capacity.csv (file_1_view_0), indexed by supplier_id.
- $c_{ij}$: transportation_costs.csv (file_2_view_0), column transportation_cost_to_{customer_id}, row supplier_id.

Index sets, parameters, and cost matrix are defined by the exact identifiers and source order in the current data. No data is omitted or aggregated. All constraints and variable domains are as specified in the user query.