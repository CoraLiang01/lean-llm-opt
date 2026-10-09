Let:
- I = set of suppliers (facilities): {MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES}
- J = set of stores (customers): {Customer_1 (CLARINDA), Customer_2 (FORT MADISON), Customer_3 (SIOUX CITY), Customer_4 (TOLEDO), Customer_5 (BANCROFT)}

Parameters:
- Fixed cost for each supplier i ∈ I:
    - f = [96.58 (MOUNT AYR), 94.06 (WAUKEE), 94.37 (WAVERLY), 82.88 (PELLA), 94.96 (DES MOINES)]
- Demand for each store j ∈ J:
    - d = [2397 (Customer_1), 1889 (Customer_2), 2518 (Customer_3), 3218 (Customer_4), 1813 (Customer_5)]
- Transportation cost per unit from supplier i to store j (c_{ij}), as a 5x5 matrix (rows: suppliers, columns: stores):

\[
C = \begin{bmatrix}
694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
15.13 & 1.50 & 1.43 & 27.88 & 90.69 \\
2.34 & 349.34 & 246.60 & 41.30 & 78.73 \\
1181.60 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
1030.80 & 43.48 & 932.43 & 55.39 & 103.84 \\
\end{bmatrix}
\]

Decision variables:
- \( y_i \in \{0,1\} \): 1 if supplier i is open, 0 otherwise
- \( x_{ij} \geq 0 \): quantity supplied from supplier i to store j

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{Subject to:} \quad & \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
& x_{ij} \leq d_j y_i \quad \forall i \in I, \forall j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I, \forall j \in J \\
\end{align*}
\]

Where:
- \( f = [96.58, 94.06, 94.37, 82.88, 94.96] \)
- \( d = [2397, 1889, 2518, 3218, 1813] \)
- \( C \) as above, with explicit mapping:
    - Row 1: MOUNT AYR to all customers
    - Row 2: WAUKEE to all customers
    - Row 3: WAVERLY to all customers
    - Row 4: PELLA to all customers
    - Row 5: DES MOINES to all customers
    - Columns: Customer_1 (CLARINDA), Customer_2 (FORT MADISON), Customer_3 (SIOUX CITY), Customer_4 (TOLEDO), Customer_5 (BANCROFT)

This model determines which suppliers to activate (y_i) and how much each supplier should ship to each store (x_{ij}) to meet all store demands at minimum total cost, including both fixed and transportation costs.