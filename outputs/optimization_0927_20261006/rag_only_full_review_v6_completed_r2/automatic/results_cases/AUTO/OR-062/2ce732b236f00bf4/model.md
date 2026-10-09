Sets:
- Suppliers I = {1: MOUNT AYR, 2: WAUKEE, 3: WAVERLY, 4: PELLA, 5: DES MOINES}
- Stores J = {1: CLARINDA, 2: FORT MADISON, 3: SIOUX CITY, 4: TOLEDO, 5: BANCROFT}

Parameters:
- Fixed costs: f = [96.58, 94.06, 94.37, 82.88, 94.96]
- Transportation costs:
\[
c = \begin{bmatrix}
694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
15.13 & 1.5 & 1.43 & 27.88 & 90.69 \\
2.34 & 349.34 & 246.6 & 41.3 & 78.73 \\
1181.6 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
1030.8 & 43.48 & 932.43 & 55.39 & 103.84 \\
\end{bmatrix}
\]
- Demands: d = [2397, 1889, 2518, 3218, 1813]

Decision variables:
- y_i ∈ {0,1} for i=1..5 (supplier activation)
- x_{ij} ≥ 0 for i=1..5, j=1..5 (quantity shipped)

Model:
\[
\min \sum_{i=1}^5 f_i y_i + \sum_{i=1}^5 \sum_{j=1}^5 c_{ij} x_{ij}
\]
subject to
\[
\sum_{i=1}^5 x_{ij} = d_j \quad \forall j=1..5
\]
\[
x_{ij} \leq d_j y_i \quad \forall i=1..5, j=1..5
\]
\[
y_i \in \{0,1\} \quad \forall i=1..5
\]
\[
x_{ij} \geq 0 \quad \forall i=1..5, j=1..5
\]

All parameters (fixed costs, transportation costs, demands) are as listed above. This model determines which suppliers to activate and how to allocate shipments to minimize total cost while meeting all store demands.