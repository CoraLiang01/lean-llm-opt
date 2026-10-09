Mathematical Model (Uncapacitated Facility Location Problem with Capacities):

Sets:
- Let \( I \) be the set of warehouses, indexed by \( i \), where \( I = \{\text{W1}, \text{W2}, \ldots, \text{W10}\} \) (from file_1_view_0["Warehouse ID"]).
- Let \( J \) be the set of customers, indexed by \( j \), where \( J = \{\text{C1}, \text{C2}, \ldots, \text{C20}\} \) (from file_2_view_0["Customer ID"]).

Parameters:
- \( f_i \): Fixed cost to open warehouse \( i \) (from file_1_view_0["Fixed_Cost"]).
- \( s_i \): Capacity of warehouse \( i \) (from file_1_view_0["Capacity"]).
- \( d_j \): Demand of customer \( j \) (from file_2_view_0["Demand"]).
- \( c_{ij} \): Transportation cost per unit from warehouse \( i \) to customer \( j \) (from file_0_view_0, row "Warehouse ID" = \( i \), column \( j \)).

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if warehouse \( i \) is opened, 0 otherwise.
- \( x_{ij} \geq 0 \): Amount of customer \( j \)'s demand served from warehouse \( i \).

Objective:
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each customer:
\[
\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
\]
2. Warehouse capacity:
\[
\sum_{j \in J} x_{ij} \leq s_i y_i \quad \forall i \in I
\]
3. Non-negativity and binary constraints:
\[
x_{ij} \geq 0 \quad \forall i \in I, j \in J
\]
\[
y_i \in \{0,1\} \quad \forall i \in I
\]

Data Mapping:
- \( I \): file_1_view_0["Warehouse ID"]
- \( J \): file_2_view_0["Customer ID"]
- \( f_i \): file_1_view_0["Fixed_Cost"], indexed by "Warehouse ID"
- \( s_i \): file_1_view_0["Capacity"], indexed by "Warehouse ID"
- \( d_j \): file_2_view_0["Demand"], indexed by "Customer ID"
- \( c_{ij} \): file_0_view_0, row "Warehouse ID" = \( i \), column \( j \) (column names "C1", ..., "C20")

Variables:
- \( y_i \): Binary, for each \( i \in I \)
- \( x_{ij} \): Continuous, for each \( i \in I, j \in J \)

All index sets, parameters, and constraints are mapped directly to the provided CSV data.