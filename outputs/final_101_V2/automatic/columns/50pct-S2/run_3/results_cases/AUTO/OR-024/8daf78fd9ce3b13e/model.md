##### Decision Variables

$x_{ij} \geq 0$: Quantity of goods shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
$y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise.

##### Parameters

- Warehouses (Facilities): $I = \{\text{S1}, \text{S2}, \text{S3}\}$
- Musicians/Bands (Customers): $J = \{\text{C1}, \text{C2}, \text{C3}\}$

- Demand:
  - $d_{\text{C1}} = 1083$
  - $d_{\text{C2}} = 776$
  - $d_{\text{C3}} = 16214$

- Fixed Costs:
  - $f_{\text{S1}} = 102.33$
  - $f_{\text{S2}} = 94.92$
  - $f_{\text{S3}} = 91.83$

- Transportation Costs $c_{ij}$:

|         | C1      | C2      | C3     |
|---------|---------|---------|--------|
| S1      | 1506.22 | 70.9    | 8.44   |
| S2      | 1732.65 | 1780.72 | 567.44 |
| S3      | 115.66  | 100.76  | 64.68  |

- $M = \sum_{j \in J} d_j = 1083 + 776 + 16214 = 18073$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand Satisfaction** (each musician/band receives exactly their demand):

   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Warehouse Activation** (no shipments from inactive warehouses):

   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable Domains**:

   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Complete Model (with all parameters)

Let $I = \{\text{S1}, \text{S2}, \text{S3}\}$, $J = \{\text{C1}, \text{C2}, \text{C3}\}$.

\[
\begin{align*}
\min\ & 1506.22\,x_{\text{S1},\text{C1}} + 70.9\,x_{\text{S1},\text{C2}} + 8.44\,x_{\text{S1},\text{C3}} \\
     & + 1732.65\,x_{\text{S2},\text{C1}} + 1780.72\,x_{\text{S2},\text{C2}} + 567.44\,x_{\text{S2},\text{C3}} \\
     & + 115.66\,x_{\text{S3},\text{C1}} + 100.76\,x_{\text{S3},\text{C2}} + 64.68\,x_{\text{S3},\text{C3}} \\
     & + 102.33\,y_{\text{S1}} + 94.92\,y_{\text{S2}} + 91.83\,y_{\text{S3}} \\
\text{s.t.}\quad
& x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} = 1083 \\
& x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} = 776 \\
& x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} = 16214 \\
& x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + x_{\text{S1},\text{C3}} \leq 18073\,y_{\text{S1}} \\
& x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + x_{\text{S2},\text{C3}} \leq 18073\,y_{\text{S2}} \\
& x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + x_{\text{S3},\text{C3}} \leq 18073\,y_{\text{S3}} \\
& x_{ij} \geq 0,\quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\},\quad \forall i \in I
\end{align*}
\]

All parameters, vectors, and matrices are included as retrieved from the CSV files.