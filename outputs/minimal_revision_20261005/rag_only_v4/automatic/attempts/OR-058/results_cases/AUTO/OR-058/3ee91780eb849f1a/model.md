Mathematical Model

Sets:
- Let \( F \) be the set of suppliers, indexed by \( i \), where \( F = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}\} \) (from file_1_view_0.Unnamed: 0).
- Let \( C \) be the set of stores, indexed by \( j \), where \( C = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}\} \) (from file_0_view_0.customer).

Parameters:
- \( f_i \): Fixed cost of opening supplier \( i \) (from file_1_view_0.fixed_costs, indexed by Unnamed: 0).
- \( t_{ij} \): Transportation cost per unit from supplier \( i \) to store \( j \) (from file_2_view_0, row Unnamed: 0, column \( j \)).
- \( d_j \): Demand at store \( j \) (from file_0_view_0.demand, indexed by customer).

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if supplier \( i \) is open, 0 otherwise.
- \( x_{ij} \geq 0 \): Quantity supplied from supplier \( i \) to store \( j \).

Objective:
\[
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} t_{ij} x_{ij}
\]

Constraints:
1. Demand satisfaction at each store:
\[
\sum_{i \in F} x_{ij} = d_j \quad \forall j \in C
\]
2. Supply only from open suppliers:
\[
x_{ij} \leq d_j y_i \quad \forall i \in F,\, j \in C
\]
3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in F
\]
\[
x_{ij} \geq 0 \quad \forall i \in F,\, j \in C
\]

Data Mapping

- \( F = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}\} \) from file_1_view_0.Unnamed: 0
- \( C = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}\} \) from file_0_view_0.customer
- \( f_i \) from file_1_view_0.fixed_costs, indexed by Unnamed: 0
- \( t_{ij} \) from file_2_view_0, row Unnamed: 0, column \( j \)
- \( d_j \) from file_0_view_0.demand, indexed by customer