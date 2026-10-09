##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the source data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer group $j \in J$ (from file_0_view_0, column "demand").
- $s_i$: supply capacity of distribution center $i \in I$ (from file_1_view_0, column "supply_capacity").
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from file_2_view_0, columns "transportation_cost_to_Ck" for each $j$).

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

- $I$ (distribution centers): All "supplier_id" in file_1_view_0 and file_2_view_0, in source order.
- $J$ (customer groups): All "customer_id" in file_0_view_0 and corresponding columns in file_2_view_0, in source order.
- $d_j$: file_0_view_0, column "demand", indexed by "customer_id".
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "supplier_id".
- $c_{ij}$: file_2_view_0, row "supplier_id" $i$, column "transportation_cost_to_Ck" for customer $j$.

Index sets, parameters, and cost matrix are defined exactly as in the current Observation, preserving all identifiers and source order.