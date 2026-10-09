Mathematical Model:

Sets:
- Let \( S \) be the set of warehouses, indexed by \( s \), where \( S = \{\text{S1}, \text{S2}, \ldots, \text{S10}\} \) (from file_1_view_0."Unnamed: 0").
- Let \( C \) be the set of customers (stores), indexed by \( c \), where \( C = \{\text{C1}, \text{C2}, \ldots, \text{C10}\} \) (from file_0_view_0."customer").

Parameters:
- \( d_c \): Demand of customer \( c \) (from file_0_view_0."demand").
- \( u_s \): Supply capacity of warehouse \( s \) (from file_1_view_0."supply_capacity").
- \( t_{s,c} \): Transportation cost per unit from warehouse \( s \) to customer \( c \) (from file_2_view_0, row "Unnamed: 0" = \( s \), column \( c \)).

Decision Variables:
- \( x_{s,c} \geq 0 \): Number of units shipped from warehouse \( s \) to customer \( c \).

Objective:
\[
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
\]

Subject to:
1. Demand satisfaction for each customer:
\[
\sum_{s \in S} x_{s,c} = d_c \quad \forall c \in C
\]
2. Supply capacity for each warehouse:
\[
\sum_{c \in C} x_{s,c} \leq u_s \quad \forall s \in S
\]
3. Non-negativity:
\[
x_{s,c} \geq 0 \quad \forall s \in S,\, c \in C
\]

Data Mapping:
- \( S \): All values in file_1_view_0."Unnamed: 0"
- \( C \): All values in file_0_view_0."customer"
- \( d_c \): file_0_view_0."demand", with \( c \) = file_0_view_0."customer"
- \( u_s \): file_1_view_0."supply_capacity", with \( s \) = file_1_view_0."Unnamed: 0"
- \( t_{s,c} \): file_2_view_0, row "Unnamed: 0" = \( s \), column \( c \)

Variable Domains:
- \( x_{s,c} \geq 0 \), continuous, for all \( s \in S, c \in C \)

All indices, parameters, and coefficients are bound exactly to the current source data as described above.