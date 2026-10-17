Let:

- $F$ = set of suppliers = {MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES}
- $S$ = set of stores = {CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT}
- $D$ = set of customers (stores with demand) = {Customer_1, Customer_2, Customer_3, Customer_4, Customer_5}
- $d_k$ = demand for Customer $k$ (from demand.csv)
- $f_i$ = fixed cost for supplier $i$ (from fixed_cost.csv)
- $c_{ij}$ = transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv)
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to store $j$

#### Data

- Demands (in source order):

  - Customer_1: $d_1 = 2397$
  - Customer_2: $d_2 = 1889$
  - Customer_3: $d_3 = 2518$
  - Customer_4: $d_4 = 3218$
  - Customer_5: $d_5 = 1813$

- Suppliers and their fixed costs (in source order):

  - MOUNT AYR: $f_1 = 96.58$
  - WAUKEE: $f_2 = 94.06$
  - WAVERLY: $f_3 = 94.37$
  - PELLA: $f_4 = 82.88$
  - DES MOINES: $f_5 = 94.96$

- Transportation costs $c_{ij}$ (supplier $i$ to store $j$):

| Supplier \ Store | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|------------------|----------|--------------|------------|--------|----------|
| MOUNT AYR        | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE           | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY          | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA            | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES       | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

#### Mathematical Model

Minimize total cost:
$$
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij}
$$

Subject to:

1. **Demand satisfaction:**  
   For each store $j$ (corresponding to Customer $k$ in source order), the total supply received equals its demand:
   $$
   \sum_{i \in F} x_{ij} = d_k, \quad \forall j \in S, \text{ where } j \text{ is the $k$-th store in source order}
   $$
   Specifically:
   - $\sum_{i} x_{i,\text{CLARINDA}} = 2397$
   - $\sum_{i} x_{i,\text{FORT MADISON}} = 1889$
   - $\sum_{i} x_{i,\text{SIOUX CITY}} = 2518$
   - $\sum_{i} x_{i,\text{TOLEDO}} = 3218$
   - $\sum_{i} x_{i,\text{BANCROFT}} = 1813$

2. **Supplier activation:**  
   A supplier must be open to supply any store:
   $$
   x_{ij} \leq M_{ij} y_i, \quad \forall i \in F, \forall j \in S
   $$
   Where $M_{ij}$ is a sufficiently large constant, e.g., $M_{ij} = \max_k d_k$ (here, $M_{ij} = 3218$).

3. **Variable domains:**
   $$
   x_{ij} \geq 0, \quad \forall i \in F, \forall j \in S
   $$
   $$
   y_i \in \{0,1\}, \quad \forall i \in F
   $$

#### Explicit Formulation with Data

Let $F = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$  
Let $S = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$

**Decision variables:**
- $y_{\text{MOUNT AYR}}, y_{\text{WAUKEE}}, y_{\text{WAVERLY}}, y_{\text{PELLA}}, y_{\text{DES MOINES}} \in \{0,1\}$
- $x_{i,j} \geq 0$ for all $i \in F$, $j \in S$

**Objective:**
\[
\begin{align*}
\min \quad & 96.58\, y_{\text{MOUNT AYR}} + 94.06\, y_{\text{WAUKEE}} + 94.37\, y_{\text{WAVERLY}} + 82.88\, y_{\text{PELLA}} + 94.96\, y_{\text{DES MOINES}} \\
& + 694.68\, x_{\text{MOUNT AYR},\text{CLARINDA}} + 17.48\, x_{\text{MOUNT AYR},\text{FORT MADISON}} + 20.07\, x_{\text{MOUNT AYR},\text{SIOUX CITY}} + 199.02\, x_{\text{MOUNT AYR},\text{TOLEDO}} + 1685.53\, x_{\text{MOUNT AYR},\text{BANCROFT}} \\
& + 15.13\, x_{\text{WAUKEE},\text{CLARINDA}} + 1.50\, x_{\text{WAUKEE},\text{FORT MADISON}} + 1.43\, x_{\text{WAUKEE},\text{SIOUX CITY}} + 27.88\, x_{\text{WAUKEE},\text{TOLEDO}} + 90.69\, x_{\text{WAUKEE},\text{BANCROFT}} \\
& + 2.34\, x_{\text{WAVERLY},\text{CLARINDA}} + 349.34\, x_{\text{WAVERLY},\text{FORT MADISON}} + 246.60\, x_{\text{WAVERLY},\text{SIOUX CITY}} + 41.30\, x_{\text{WAVERLY},\text{TOLEDO}} + 78.73\, x_{\text{WAVERLY},\text{BANCROFT}} \\
& + 1181.60\, x_{\text{PELLA},\text{CLARINDA}} + 1458.53\, x_{\text{PELLA},\text{FORT MADISON}} + 1646.36\, x_{\text{PELLA},\text{SIOUX CITY}} + 1924.55\, x_{\text{PELLA},\text{TOLEDO}} + 38.93\, x_{\text{PELLA},\text{BANCROFT}} \\
& + 1030.80\, x_{\text{DES MOINES},\text{CLARINDA}} + 43.48\, x_{\text{DES MOINES},\text{FORT MADISON}} + 932.43\, x_{\text{DES MOINES},\text{SIOUX CITY}} + 55.39\, x_{\text{DES MOINES},\text{TOLEDO}} + 103.84\, x_{\text{DES MOINES},\text{BANCROFT}}
\end{align*}
\]

**Subject to:**

- Demand constraints:
  \[
  \begin{align*}
  x_{\text{MOUNT AYR},\text{CLARINDA}} + x_{\text{WAUKEE},\text{CLARINDA}} + x_{\text{WAVERLY},\text{CLARINDA}} + x_{\text{PELLA},\text{CLARINDA}} + x_{\text{DES MOINES},\text{CLARINDA}} &= 2397 \\
  x_{\text{MOUNT AYR},\text{FORT MADISON}} + x_{\text{WAUKEE},\text{FORT MADISON}} + x_{\text{WAVERLY},\text{FORT MADISON}} + x_{\text{PELLA},\text{FORT MADISON}} + x_{\text{DES MOINES},\text{FORT MADISON}} &= 1889 \\
  x_{\text{MOUNT AYR},\text{SIOUX CITY}} + x_{\text{WAUKEE},\text{SIOUX CITY}} + x_{\text{WAVERLY},\text{SIOUX CITY}} + x_{\text{PELLA},\text{SIOUX CITY}} + x_{\text{DES MOINES},\text{SIOUX CITY}} &= 2518 \\
  x_{\text{MOUNT AYR},\text{TOLEDO}} + x_{\text{WAUKEE},\text{TOLEDO}} + x_{\text{WAVERLY},\text{TOLEDO}} + x_{\text{PELLA},\text{TOLEDO}} + x_{\text{DES MOINES},\text{TOLEDO}} &= 3218 \\
  x_{\text{MOUNT AYR},\text{BANCROFT}} + x_{\text{WAUKEE},\text{BANCROFT}} + x_{\text{WAVERLY},\text{BANCROFT}} + x_{\text{PELLA},\text{BANCROFT}} + x_{\text{DES MOINES},\text{BANCROFT}} &= 1813 \\
  \end{align*}
  \]

- Supplier activation constraints (for all $i \in F$, $j \in S$):
  \[
  x_{ij} \leq 3218\, y_i
  \]

- Variable domains:
  \[
  x_{ij} \geq 0 \quad \forall i \in F,\, j \in S
  \]
  \[
  y_i \in \{0,1\} \quad \forall i \in F
  \]