Let:
- I = set of suppliers (facilities): {1: MOUNT AYR, 2: WAUKEE, 3: WAVERLY, 4: PELLA, 5: DES MOINES}
- J = set of stores (customers): {1: CLARINDA, 2: FORT MADISON, 3: SIOUX CITY, 4: TOLEDO, 5: BANCROFT}

Parameters:
- Fixed costs for each supplier (vector f):
    - f = [f₁, f₂, f₃, f₄, f₅] = [96.58, 94.06, 94.37, 82.88, 94.96]
      - f₁ (MOUNT AYR): 96.58
      - f₂ (WAUKEE): 94.06
      - f₃ (WAVERLY): 94.37
      - f₄ (PELLA): 82.88
      - f₅ (DES MOINES): 94.96

- Demand for each store (vector d):
    - d = [d₁, d₂, d₃, d₄, d₅] = [2397, 1889, 2518, 3218, 1813]
      - d₁ (CLARINDA): 2397
      - d₂ (FORT MADISON): 1889
      - d₃ (SIOUX CITY): 2518
      - d₄ (TOLEDO): 3218
      - d₅ (BANCROFT): 1813

- Transportation cost per unit from supplier i to store j (matrix c):
    - c = [c_{ij}] where rows are suppliers (in order: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES) and columns are stores (in order: CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT):

\[
c = \begin{bmatrix}
694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
15.13 & 1.50 & 1.43 & 27.88 & 90.69 \\
2.34 & 349.34 & 246.60 & 41.30 & 78.73 \\
1181.60 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
1030.80 & 43.48 & 932.43 & 55.39 & 103.84 \\
\end{bmatrix}
\]

Decision variables:
- \( y_i \in \{0,1\} \): 1 if supplier i is open, 0 otherwise, for i = 1,...,5
- \( x_{ij} \geq 0 \): quantity supplied from supplier i to store j, for i = 1,...,5; j = 1,...,5

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i=1}^{5} f_i y_i + \sum_{i=1}^{5} \sum_{j=1}^{5} c_{ij} x_{ij} \\
\text{Subject to:} \quad & \sum_{i=1}^{5} x_{ij} = d_j \quad \forall j = 1,...,5 \\
& x_{ij} \leq d_j y_i \quad \forall i = 1,...,5; \; j = 1,...,5 \\
& y_i \in \{0,1\} \quad \forall i = 1,...,5 \\
& x_{ij} \geq 0 \quad \forall i = 1,...,5; \; j = 1,...,5 \\
\end{align*}
\]

Where:
- \( f_i \) is the fixed cost for supplier i (see vector above)
- \( c_{ij} \) is the transportation cost per unit from supplier i to store j (see matrix above)
- \( d_j \) is the demand for store j (see vector above)

All parameters (vectors and matrices) are explicitly stated as retrieved from the CSV files. The objective is to minimize the total cost (fixed + transportation) while meeting all store demands and only allowing supply from open suppliers.