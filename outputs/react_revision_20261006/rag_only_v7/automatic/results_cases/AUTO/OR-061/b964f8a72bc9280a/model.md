Mathematical Model:

Sets:
- Let \( F \) be the set of suppliers, indexed by \( i \), with IDs from file_1_view_0["Unnamed: 0"] and file_2_view_0["Unnamed: 0"].
- Let \( C \) be the set of branches (customers), indexed by \( j \), with IDs from file_0_view_0["customer"] and file_2_view_0 columns ["C1", "C2", "C3", "C4", "C5"].

Parameters:
- \( f_i \): Fixed cost of opening supplier \( i \), from file_1_view_0["fixed_costs"].
- \( d_j \): Demand of branch \( j \), from file_0_view_0["demand"].
- \( t_{ij} \): Transportation cost per unit from supplier \( i \) to branch \( j \), from file_2_view_0, with row index file_2_view_0["Unnamed: 0"] and column index file_2_view_0 columns ["C1", "C2", "C3", "C4", "C5"].

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if supplier \( i \) is open, 0 otherwise.
- \( x_{ij} \geq 0 \): Quantity of goods supplied from supplier \( i \) to branch \( j \).

Objective:
\[
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} t_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each branch:
\[
\sum_{i \in F} x_{ij} = d_j \quad \forall j \in C
\]
2. Supply only from open suppliers:
\[
x_{ij} \leq d_j y_i \quad \forall i \in F, \forall j \in C
\]
3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in F
\]
\[
x_{ij} \geq 0 \quad \forall i \in F, \forall j \in C
\]

Data Mapping:
- \( F \): All supplier IDs from file_1_view_0["Unnamed: 0"] and file_2_view_0["Unnamed: 0"].
- \( C \): All branch/customer IDs from file_0_view_0["customer"] and file_2_view_0 columns ["C1", "C2", "C3", "C4", "C5"].
- \( f_i \): file_1_view_0["fixed_costs"], indexed by file_1_view_0["Unnamed: 0"].
- \( d_j \): file_0_view_0["demand"], indexed by file_0_view_0["customer"].
- \( t_{ij} \): file_2_view_0, with row index file_2_view_0["Unnamed: 0"] (supplier) and column index file_2_view_0 columns ["C1", "C2", "C3", "C4", "C5"] (branch/customer).

All indices, parameters, and constraints are mapped exactly to the current CSV data as described above.