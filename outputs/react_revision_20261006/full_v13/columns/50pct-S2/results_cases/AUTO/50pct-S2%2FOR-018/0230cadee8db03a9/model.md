##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the source data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer group $j$ (from file_0_view_0, column "demand").
- $s_i$: supply capacity of distribution center $i$ (from file_1_view_0, column "supply_capacity").
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from file_2_view_0, columns "transportation_cost_to_Ck", $k=1,\ldots,12$).

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
- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$ (from file_1_view_0, column "supplier_id")
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$ (from file_0_view_0, column "customer_id")

##### Data Mapping

- $d_j$: file_0_view_0, column "demand", indexed by "customer_id"
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "supplier_id"
- $c_{ij}$: file_2_view_0, row "supplier_id" $i$, column "transportation_cost_to_Ck" where $j$ = Ck
- $x_{ij}$: decision variable for each $(i,j)$ pair, $i$ from file_1_view_0 "supplier_id", $j$ from file_0_view_0 "customer_id"

All index sets, parameters, and constraints are defined directly from the current source data, preserving all identifiers and bounds.