##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (operational), 0 otherwise.

##### Sets

- $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$ (Suppliers)
- $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$ (Stores)

##### Parameters

- Demands (from demand.csv):

| Store         | Demand |
|---------------|--------|
| Customer_1    | 2397   |
| Customer_2    | 1889   |
| Customer_3    | 2518   |
| Customer_4    | 3218   |
| Customer_5    | 1813   |

Let us map stores to the store names in the transportation cost matrix. Since the transportation cost matrix columns are: CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT, we will assume:

- $d_{\text{CLARINDA}} = 2397$
- $d_{\text{FORT MADISON}} = 1889$
- $d_{\text{SIOUX CITY}} = 2518$
- $d_{\text{TOLEDO}} = 3218$
- $d_{\text{BANCROFT}} = 1813$

- Fixed costs (from fixed_cost.csv):

| Supplier      | Fixed Cost |
|---------------|------------|
| MOUNT AYR     | 96.58      |
| WAUKEE        | 94.06      |
| WAVERLY       | 94.37      |
| PELLA         | 82.88      |
| DES MOINES    | 94.96      |

- Transportation costs (from transportation_costs.csv):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

Let $c_{ij}$ denote the transportation cost per unit from supplier $i$ to store $j$.

- $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:** Each store's demand must be met.
   \[
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   \]

2. **Supplier activation:** No shipments from inactive suppliers.
   \[
   \sum_{j \in J} x_{ij} \leq M y_i \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Parameter Tables

- **Suppliers ($I$):** MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- **Stores ($J$):** CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT

- **Demands ($d_j$):**

| Store         | Demand |
|---------------|--------|
| CLARINDA      | 2397   |
| FORT MADISON  | 1889   |
| SIOUX CITY    | 2518   |
| TOLEDO        | 3218   |
| BANCROFT      | 1813   |

- **Fixed Costs ($f_i$):**

| Supplier      | Fixed Cost |
|---------------|------------|
| MOUNT AYR     | 96.58      |
| WAUKEE        | 94.06      |
| WAVERLY       | 94.37      |
| PELLA         | 82.88      |
| DES MOINES    | 94.96      |

- **Transportation Costs ($c_{ij}$):**

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

- **Big-M parameter:** $M = 11835$

---

**Complete Mathematical Model:**

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]

with all parameters as listed above.