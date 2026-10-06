Mathematical Model

Sets:
- Let \( F \) be the set of suppliers, indexed by \( i \), where \( F = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\} \) (from file_1_view_0 and file_2_view_0, column "Unnamed: 0").
- Let \( S \) be the set of stores, indexed by \( j \), where \( S = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\} \) (from file_2_view_0, columns).

Parameters:
- \( \text{fixed\_cost}_i \): Fixed cost to open supplier \( i \). Data Mapping: file_1_view_0, columns "Unnamed: 0" (supplier), "fixed_costs".
- \( \text{demand}_j \): Demand at store \( j \). Data Mapping: file_0_view_0, columns "Customer" (store), "demand".
- \( \text{trans\_cost}_{ij} \): Transportation cost per unit from supplier \( i \) to store \( j \). Data Mapping: file_2_view_0, rows "Unnamed: 0" (supplier), columns (store names).

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if supplier \( i \) is open, 0 otherwise.
- \( x_{ij} \geq 0 \): Quantity supplied from supplier \( i \) to store \( j \).

Objective:
\[
\min \sum_{i \in F} \text{fixed\_cost}_i \cdot y_i + \sum_{i \in F} \sum_{j \in S} \text{trans\_cost}_{ij} \cdot x_{ij}
\]

Constraints:
1. Demand satisfaction at each store:
\[
\forall j \in S: \quad \sum_{i \in F} x_{ij} = \text{demand}_j
\]
  (Data Mapping: file_0_view_0, "Customer" = \( j \), "demand")

2. Linking supplier activation to shipments:
\[
\forall i \in F, \forall j \in S: \quad x_{ij} \leq M_{ij} \cdot y_i
\]
where \( M_{ij} \) is a sufficiently large constant, e.g., \( M_{ij} = \text{demand}_j \).
  (Data Mapping: file_0_view_0, "Customer" = \( j \), "demand")

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in F
\]
\[
x_{ij} \geq 0 \quad \forall i \in F, j \in S
\]

Data Mapping Summary:
- Suppliers (\( F \)): file_1_view_0, "Unnamed: 0"; file_2_view_0, "Unnamed: 0"
- Stores (\( S \)): file_2_view_0, columns "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"
- \( \text{fixed\_cost}_i \): file_1_view_0, "fixed_costs"
- \( \text{demand}_j \): file_0_view_0, "demand"
- \( \text{trans\_cost}_{ij} \): file_2_view_0, row "Unnamed: 0" = \( i \), column \( j \)

Model Summary:
\[
\begin{align*}
\min_{x, y} \quad & \sum_{i \in F} \text{fixed\_cost}_i \cdot y_i + \sum_{i \in F} \sum_{j \in S} \text{trans\_cost}_{ij} \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{i \in F} x_{ij} = \text{demand}_j \quad \forall j \in S \\
& x_{ij} \leq \text{demand}_j \cdot y_i \quad \forall i \in F, j \in S \\
& y_i \in \{0,1\} \quad \forall i \in F \\
& x_{ij} \geq 0 \quad \forall i \in F, j \in S
\end{align*}
\]

All sets and parameters are mapped directly to the provided CSV data as specified above.