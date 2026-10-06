Mathematical Model

Sets:
- Let \( F \) be the set of suppliers, indexed by \( i \), where \( F = \{\text{S1}, \text{S2}, \ldots, \text{S12}\} \) from file_1_view_0.Unnamed: 0.
- Let \( C \) be the set of supermarkets, indexed by \( j \), where \( C = \{\text{C1}, \text{C2}, \ldots, \text{C12}\} \) from file_0_view_0.customer.

Parameters:
- \( f_i \): Fixed cost of opening supplier \( i \), from file_1_view_0.fixed_costs.
- \( t_{ij} \): Transportation cost per unit from supplier \( i \) to supermarket \( j \), from file_2_view_0, row Unnamed: 0 = \( i \), column \( j \).
- \( d_j \): Demand at supermarket \( j \), from file_0_view_0.demand.

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if supplier \( i \) is open, 0 otherwise.
- \( x_{ij} \geq 0 \): Quantity supplied from supplier \( i \) to supermarket \( j \).

Objective:
\[
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} t_{ij} x_{ij}
\]

Subject to:

1. Demand satisfaction at each supermarket:
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

- \( F = \) all values in file_1_view_0.Unnamed: 0
- \( C = \) all values in file_0_view_0.customer
- \( f_i = \) file_1_view_0.fixed_costs, indexed by Unnamed: 0
- \( t_{ij} = \) file_2_view_0, row Unnamed: 0 = \( i \), column \( j \)
- \( d_j = \) file_0_view_0.demand, indexed by customer

This model ensures all supermarket demands are met at minimum total cost, with suppliers incurring fixed costs only if opened, and transportation costs proportional to the allocation.