Mathematical Model

Sets:
- Let \( F \) be the set of warehouses, indexed by \( i \), where \( F = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}\} \) (from file_1_view_0.Unnamed: 0).
- Let \( C \) be the set of musicians/bands (customers), indexed by \( j \), where \( C = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}\} \) (from file_0_view_0.customer).

Parameters:
- \( f_i \): Fixed cost to open warehouse \( i \). Data Mapping: file_1_view_0.fixed_costs, indexed by file_1_view_0.Unnamed: 0.
- \( t_{ij} \): Transportation cost per unit from warehouse \( i \) to customer \( j \). Data Mapping: file_2_view_0, row index file_2_view_0.Unnamed: 0, column index file_2_view_0.[C1,...,C7].
- \( d_j \): Demand of customer \( j \). Data Mapping: file_0_view_0.demand, indexed by file_0_view_0.customer.

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if warehouse \( i \) is open, 0 otherwise.
- \( x_{ij} \geq 0 \): Quantity supplied from warehouse \( i \) to customer \( j \).

Objective:
\[
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} t_{ij} x_{ij}
\]

Subject to:

1. Demand satisfaction for each customer:
\[
\sum_{i \in F} x_{ij} = d_j \quad \forall j \in C
\]

2. Linking supply to warehouse activation:
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

Data Mapping Summary:
- \( F \): file_1_view_0.Unnamed: 0
- \( C \): file_0_view_0.customer
- \( f_i \): file_1_view_0.fixed_costs, indexed by file_1_view_0.Unnamed: 0
- \( t_{ij} \): file_2_view_0, row index file_2_view_0.Unnamed: 0, column index file_2_view_0.[C1,...,C7]
- \( d_j \): file_0_view_0.demand, indexed by file_0_view_0.customer

This model determines which warehouses to open and how much each should supply to each customer to minimize total fixed and transportation costs, ensuring all customer demands are met.