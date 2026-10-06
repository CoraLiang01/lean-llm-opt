Let:
- I = set of suppliers = {1: MOUNT AYR, 2: WAUKEE, 3: WAVERLY, 4: PELLA, 5: DES MOINES}
- J = set of stores = {1: Customer_1 (CLARINDA), 2: Customer_2 (FORT MADISON), 3: Customer_3 (SIOUX CITY), 4: Customer_4 (TOLEDO), 5: Customer_5 (BANCROFT)}

Parameters:
- Fixed cost for each supplier \( f_i \):
    - \( f_1 = 96.58 \) (MOUNT AYR)
    - \( f_2 = 94.06 \) (WAUKEE)
    - \( f_3 = 94.37 \) (WAVERLY)
    - \( f_4 = 82.88 \) (PELLA)
    - \( f_5 = 94.96 \) (DES MOINES)
- Demand for each store \( d_j \):
    - \( d_1 = 2397 \) (Customer_1)
    - \( d_2 = 1889 \) (Customer_2)
    - \( d_3 = 2518 \) (Customer_3)
    - \( d_4 = 3218 \) (Customer_4)
    - \( d_5 = 1813 \) (Customer_5)
- Transportation cost per unit \( c_{ij} \) (from supplier i to store j):

\[
C = \begin{bmatrix}
694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
15.13 & 1.50 & 1.43 & 27.88 & 90.69 \\
2.34 & 349.34 & 246.60 & 41.30 & 78.73 \\
1181.60 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
1030.80 & 43.48 & 932.43 & 55.39 & 103.84 \\
\end{bmatrix}
\]
where row i corresponds to supplier i (in the order above), and column j corresponds to store j (in the order above).

Decision variables:
- \( y_i \in \{0,1\} \): 1 if supplier i is open, 0 otherwise
- \( x_{ij} \geq 0 \): quantity supplied from supplier i to store j

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i=1}^{5} f_i y_i + \sum_{i=1}^{5} \sum_{j=1}^{5} c_{ij} x_{ij} \\
\text{Subject to:} \quad & \sum_{i=1}^{5} x_{ij} = d_j \quad \forall j = 1,\ldots,5 \\
& x_{ij} \leq d_j y_i \quad \forall i = 1,\ldots,5; \; j = 1,\ldots,5 \\
& y_i \in \{0,1\} \quad \forall i = 1,\ldots,5 \\
& x_{ij} \geq 0 \quad \forall i = 1,\ldots,5; \; j = 1,\ldots,5 \\
\end{align*}
\]

Where:
- \( f = [96.58, 94.06, 94.37, 82.88, 94.96] \)
- \( d = [2397, 1889, 2518, 3218, 1813] \)
- \( C \) as above.

This model determines which suppliers to activate (incurring their fixed costs) and how much each supplier should ship to each store to meet all store demands at minimum total cost (fixed + transportation). All parameters (vectors and matrices) are explicitly stated above.