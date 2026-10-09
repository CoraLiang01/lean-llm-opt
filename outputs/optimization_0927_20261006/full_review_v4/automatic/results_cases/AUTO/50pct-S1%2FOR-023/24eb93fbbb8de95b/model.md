##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (operational), 0 otherwise.

##### Parameters

- Suppliers $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- Stores $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$
- Store demands:
  - $\text{Customer}_1$: $2397$
  - $\text{Customer}_2$: $1889$
  - $\text{Customer}_3$: $2518$
  - $\text{Customer}_4$: $3218$
  - $\text{Customer}_5$: $1813$
- Store mapping (for transportation costs):  
  Let us map the customers to stores as follows (since demand.csv lists 5 customers and transportation_costs.csv lists 5 stores):
  - $\text{Customer}_1 \rightarrow \text{CLARINDA}$
  - $\text{Customer}_2 \rightarrow \text{FORT MADISON}$
  - $\text{Customer}_3 \rightarrow \text{SIOUX CITY}$
  - $\text{Customer}_4 \rightarrow \text{TOLEDO}$
  - $\text{Customer}_5 \rightarrow \text{BANCROFT}$
- Demands:
  - $d_{\text{CLARINDA}} = 2397$
  - $d_{\text{FORT MADISON}} = 1889$
  - $d_{\text{SIOUX CITY}} = 2518$
  - $d_{\text{TOLEDO}} = 3218$
  - $d_{\text{BANCROFT}} = 1813$
- Supplier fixed costs:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$
- Transportation costs $c_{ij}$ (supplier $i$, store $j$):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

- $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
2. **Supplier activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### All Parameters (Vectors and Matrices)

- $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$
- $d = [2397, 1889, 2518, 3218, 1813]$ (ordered as above)
- $f = [96.58, 94.06, 94.37, 82.88, 94.96]$ (ordered as above)
- $C =$
\[
\begin{bmatrix}
694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
15.13 & 1.5 & 1.43 & 27.88 & 90.69 \\
2.34 & 349.34 & 246.6 & 41.3 & 78.73 \\
1181.6 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
1030.8 & 43.48 & 932.43 & 55.39 & 103.84 \\
\end{bmatrix}
\]
(rows: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES; columns: CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT)

- $M = 11835$

---

This is the complete mathematical model as required.