##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i \in I$ to store (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{\text{F1 (MOUNT AYR)},\ \text{F2 (WAUKEE)},\ \text{F3 (WAVERLY)},\ \text{F4 (PELLA)},\ \text{F5 (DES MOINES)}\}$
- $J = \{\text{S1 (CLARINDA)},\ \text{S2 (FORT MADISON)},\ \text{S3 (SIOUX CITY)},\ \text{S4 (TOLEDO)},\ \text{S5 (BANCROFT)}\}$

- Fixed costs ($f_i$):

  - $f_{\text{F1}} = 96.58$
  - $f_{\text{F2}} = 94.06$
  - $f_{\text{F3}} = 94.37$
  - $f_{\text{F4}} = 82.88$
  - $f_{\text{F5}} = 94.96$

- Transportation costs per unit ($c_{ij}$):

  |            | S1 (CLARINDA) | S2 (FORT MADISON) | S3 (SIOUX CITY) | S4 (TOLEDO) | S5 (BANCROFT) |
  |------------|---------------|-------------------|-----------------|-------------|---------------|
  | F1 (MOUNT AYR)    | 694.68        | 17.48            | 20.07           | 199.02      | 1685.53        |
  | F2 (WAUKEE)       | 15.13         | 1.50             | 1.43            | 27.88       | 90.69          |
  | F3 (WAVERLY)      | 2.34          | 349.34           | 246.60          | 41.30       | 78.73          |
  | F4 (PELLA)        | 1181.60       | 1458.53          | 1646.36         | 1924.55     | 38.93          |
  | F5 (DES MOINES)   | 1030.80       | 43.48            | 932.43          | 55.39       | 103.84         |

- Demand ($d_j$):

  - $d_{\text{S1}} = 2397$
  - $d_{\text{S2}} = 1889$
  - $d_{\text{S3}} = 2518$
  - $d_{\text{S4}} = 3218$
  - $d_{\text{S5}} = 1813$

- Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (sufficiently large upper bound for each supplier).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction (each store’s demand must be met):**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation (no shipments from closed suppliers):**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Complete Model (with all parameters)

\[
\begin{align*}
\min\ & \Big[694.68\,x_{\text{F1,S1}} + 17.48\,x_{\text{F1,S2}} + 20.07\,x_{\text{F1,S3}} + 199.02\,x_{\text{F1,S4}} + 1685.53\,x_{\text{F1,S5}} \\
&+ 15.13\,x_{\text{F2,S1}} + 1.50\,x_{\text{F2,S2}} + 1.43\,x_{\text{F2,S3}} + 27.88\,x_{\text{F2,S4}} + 90.69\,x_{\text{F2,S5}} \\
&+ 2.34\,x_{\text{F3,S1}} + 349.34\,x_{\text{F3,S2}} + 246.60\,x_{\text{F3,S3}} + 41.30\,x_{\text{F3,S4}} + 78.73\,x_{\text{F3,S5}} \\
&+ 1181.60\,x_{\text{F4,S1}} + 1458.53\,x_{\text{F4,S2}} + 1646.36\,x_{\text{F4,S3}} + 1924.55\,x_{\text{F4,S4}} + 38.93\,x_{\text{F4,S5}} \\
&+ 1030.80\,x_{\text{F5,S1}} + 43.48\,x_{\text{F5,S2}} + 932.43\,x_{\text{F5,S3}} + 55.39\,x_{\text{F5,S4}} + 103.84\,x_{\text{F5,S5}} \\
&+ 96.58\,y_{\text{F1}} + 94.06\,y_{\text{F2}} + 94.37\,y_{\text{F3}} + 82.88\,y_{\text{F4}} + 94.96\,y_{\text{F5}} \Big]
\end{align*}
\]

Subject to:

\[
\begin{align*}
&x_{\text{F1,S1}} + x_{\text{F2,S1}} + x_{\text{F3,S1}} + x_{\text{F4,S1}} + x_{\text{F5,S1}} = 2397 \\
&x_{\text{F1,S2}} + x_{\text{F2,S2}} + x_{\text{F3,S2}} + x_{\text{F4,S2}} + x_{\text{F5,S2}} = 1889 \\
&x_{\text{F1,S3}} + x_{\text{F2,S3}} + x_{\text{F3,S3}} + x_{\text{F4,S3}} + x_{\text{F5,S3}} = 2518 \\
&x_{\text{F1,S4}} + x_{\text{F2,S4}} + x_{\text{F3,S4}} + x_{\text{F4,S4}} + x_{\text{F5,S4}} = 3218 \\
&x_{\text{F1,S5}} + x_{\text{F2,S5}} + x_{\text{F3,S5}} + x_{\text{F4,S5}} + x_{\text{F5,S5}} = 1813 \\
\\
&x_{\text{F1,S1}} + x_{\text{F1,S2}} + x_{\text{F1,S3}} + x_{\text{F1,S4}} + x_{\text{F1,S5}} \leq 11835\,y_{\text{F1}} \\
&x_{\text{F2,S1}} + x_{\text{F2,S2}} + x_{\text{F2,S3}} + x_{\text{F2,S4}} + x_{\text{F2,S5}} \leq 11835\,y_{\text{F2}} \\
&x_{\text{F3,S1}} + x_{\text{F3,S2}} + x_{\text{F3,S3}} + x_{\text{F3,S4}} + x_{\text{F3,S5}} \leq 11835\,y_{\text{F3}} \\
&x_{\text{F4,S1}} + x_{\text{F4,S2}} + x_{\text{F4,S3}} + x_{\text{F4,S4}} + x_{\text{F4,S5}} \leq 11835\,y_{\text{F4}} \\
&x_{\text{F5,S1}} + x_{\text{F5,S2}} + x_{\text{F5,S3}} + x_{\text{F5,S4}} + x_{\text{F5,S5}} \leq 11835\,y_{\text{F5}} \\
\\
&x_{ij} \geq 0,\quad \forall i \in I,\, j \in J \\
&y_i \in \{0,1\},\quad \forall i \in I
\end{align*}
\]

All parameters, vectors, and matrices are included as retrieved from the CSV files, with all identifiers preserved.