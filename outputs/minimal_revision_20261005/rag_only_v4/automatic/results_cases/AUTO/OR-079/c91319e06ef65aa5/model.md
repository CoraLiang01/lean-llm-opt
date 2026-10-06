Mathematical Model

Sets:
- \( I \): Set of potential factory sites, indexed by \( i \). (Source: Facility column, table_id: file_0_view_0)
- \( J \): Set of distribution centers, indexed by \( j \). (Source: Destination column, table_id: file_2_view_0)

Parameters:
- \( f_i \): Fixed cost to open factory \( i \). (Source: FixedCost, table_id: file_0_view_0)
- \( s_i \): Capacity of factory \( i \). (Source: Capacity, table_id: file_0_view_0)
- \( d_j \): Demand at distribution center \( j \). (Source: Demand, table_id: file_2_view_0)
- \( c_{ij} \): Shipping cost per unit from factory \( i \) to distribution center \( j \). (Source: file_1_view_0, row: Origin = \( i \), column: \( j \))

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if factory \( i \) is constructed, 0 otherwise.
- \( x_{ij} \geq 0 \): Amount shipped from factory \( i \) to distribution center \( j \).

Objective:
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:

1. Demand satisfaction at each distribution center:
\[
\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
\]

2. Factory capacity and open/close logic:
\[
\sum_{j \in J} x_{ij} \leq s_i y_i \quad \forall i \in I
\]

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

Data Mapping

- \( I \): All Facility values in file_0_view_0 (Facility column)
- \( J \): All Destination values in file_2_view_0 (Destination column)
- \( f_i \): file_0_view_0, FixedCost column, indexed by Facility
- \( s_i \): file_0_view_0, Capacity column, indexed by Facility
- \( d_j \): file_2_view_0, Demand column, indexed by Destination
- \( c_{ij} \): file_1_view_0, row Origin = Facility \( i \), column \( j \) (B1–B8)

All indices, parameters, and constraints are mapped directly to the provided CSV data as specified above.