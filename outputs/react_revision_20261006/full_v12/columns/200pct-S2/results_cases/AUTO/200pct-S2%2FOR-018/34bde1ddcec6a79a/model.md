#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer $j$ (from "customer_demand.csv").
- $s_i$: supply capacity of supplier $i$ (from "supply_capacity.csv").
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_costs.csv").

Sets:
- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$

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

#### Data Mapping

- $I$ (suppliers): "supplier_id" in "supply_capacity.csv" (table_id: file_1_view_0)
- $J$ (customers): "customer_id" in "customer_demand.csv" (table_id: file_0_view_0)
- $d_j$: "demand" column in "customer_demand.csv" (table_id: file_0_view_0, key: customer_id)
- $s_i$: "supply_capacity" column in "supply_capacity.csv" (table_id: file_1_view_0, key: supplier_id)
- $c_{ij}$: "transportation_cost_to_Ck" columns in "transportation_costs.csv" (table_id: file_2_view_0, row: supplier_id $i$, column: customer $j$ as mapped by relationships in the Observation)

All indices, parameters, and constraints are mapped directly to the current source data and identifiers.