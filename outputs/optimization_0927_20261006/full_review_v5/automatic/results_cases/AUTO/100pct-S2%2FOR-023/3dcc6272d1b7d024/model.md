##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (Suppliers)
- $J = \{$CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT$\}$ (Stores)
- $f_i$: Fixed cost for opening supplier $i$:

| Supplier      | Fixed Cost ($f_i$) |
|---------------|-------------------|
| MOUNT AYR     | 96.58             |
| WAUKEE        | 94.06             |
| WAVERLY       | 94.37             |
| PELLA         | 82.88             |
| DES MOINES    | 94.96             |

- $d_j$: Demand at store $j$:

| Store         | Demand ($d_j$) |
|---------------|---------------|
| Customer_1    | 2397          |
| Customer_2    | 1889          |
| Customer_3    | 2518          |
| Customer_4    | 3218          |
| Customer_5    | 1813          |

(Assuming mapping: CLARINDA = Customer_1, FORT MADISON = Customer_2, SIOUX CITY = Customer_3, TOLEDO = Customer_4, BANCROFT = Customer_5)

- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$:

| Supplier    | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|-------------|----------|--------------|------------|--------|----------|
| MOUNT AYR   | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE      | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY     | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA       | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES  | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

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
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Parameter Tables

**Suppliers ($I$):**
- MOUNT AYR
- WAUKEE
- WAVERLY
- PELLA
- DES MOINES

**Stores ($J$):**
- CLARINDA
- FORT MADISON
- SIOUX CITY
- TOLEDO
- BANCROFT

**Demands ($d_j$):**

| Store         | Demand |
|---------------|--------|
| CLARINDA      | 2397   |
| FORT MADISON  | 1889   |
| SIOUX CITY    | 2518   |
| TOLEDO        | 3218   |
| BANCROFT      | 1813   |

**Fixed Costs ($f_i$):**

| Supplier      | Fixed Cost |
|---------------|------------|
| MOUNT AYR     | 96.58      |
| WAUKEE        | 94.06      |
| WAVERLY       | 94.37      |
| PELLA         | 82.88      |
| DES MOINES    | 94.96      |

**Transportation Costs ($c_{ij}$):**

| Supplier    | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|-------------|----------|--------------|------------|--------|----------|
| MOUNT AYR   | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE      | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY     | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA       | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES  | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

**Big-M value:** $M = 11835$

---

**Complete Mathematical Model:**

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

with all parameters as listed above.