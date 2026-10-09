Mathematical Model (Uncapacitated Facility Location Problem with Capacities):

Sets:
- Let \( I \) be the set of plants, indexed by \( i \), where \( I = \{\text{F1}, \ldots, \text{F15}\} \) (from file_0_view_0.plant).
- Let \( J \) be the set of customers, indexed by \( j \), where \( J = \{\text{C1}, \ldots, \text{C15}\} \) (from file_1_view_0.customer).

Parameters:
- \( f_i \): Fixed opening cost for plant \( i \) (from file_0_view_0.fixed_cost, for each \( i \in I \)).
- \( k_i \): Capacity of plant \( i \) (from file_0_view_0.capacity, for each \( i \in I \)).
- \( d_j \): Demand of customer \( j \) (from file_1_view_0.demand, for each \( j \in J \)).
- \( c_{ij} \): Per-unit transportation cost from plant \( i \) to customer \( j \) (from file_0_view_0, column \( j \) for each \( i \in I, j \in J \)).

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if plant \( i \) is opened, 0 otherwise, for all \( i \in I \).
- \( x_{ij} \geq 0 \): Amount shipped from plant \( i \) to customer \( j \), for all \( i \in I, j \in J \).

Objective:
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each customer:
\[
\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
\]
2. Capacity constraint for each plant:
\[
\sum_{j \in J} x_{ij} \leq k_i y_i \quad \forall i \in I
\]
3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I, j \in J
\]

Data Mapping:
- \( I \): All values in file_0_view_0.plant
- \( J \): All values in file_1_view_0.customer
- \( f_i \): file_0_view_0.fixed_cost, mapped by plant \( i \)
- \( k_i \): file_0_view_0.capacity, mapped by plant \( i \)
- \( d_j \): file_1_view_0.demand, mapped by customer \( j \)
- \( c_{ij} \): file_0_view_0, column \( j \) (C1–C15), row \( i \) (plant), for all \( i \in I, j \in J \)

Variable domains and all index sets are defined by the full current data in the CSVs. No values are omitted or invented.