##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the current data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer group $j \in J$ (from "customer_demand.csv").
- $s_i$: supply capacity of distribution center $i \in I$ (from "supply_capacity.csv").
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from "transportation_costs.csv").

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

Where:
- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$

##### Data Mapping

- $I$ (distribution centers): "supplier_id" column in "supply_capacity.csv" (table_id: file_1_view_0)
- $J$ (customer groups): "customer_id" column in "customer_demand.csv" (table_id: file_0_view_0)
- $d_j$: "demand" column in "customer_demand.csv" (table_id: file_0_view_0), indexed by "customer_id"
- $s_i$: "supply_capacity" column in "supply_capacity.csv" (table_id: file_1_view_0), indexed by "supplier_id"
- $c_{ij}$: "transportation_cost_to_Ck" columns in "transportation_costs.csv" (table_id: file_2_view_0), with row index "supplier_id" and column index "customer_id" as mapped in the relationships section

All index sets, parameters, and constraints are defined directly from the current source data, preserving all identifiers and bounds.