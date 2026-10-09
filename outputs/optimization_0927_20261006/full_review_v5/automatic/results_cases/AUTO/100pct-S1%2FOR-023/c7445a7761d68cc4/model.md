##### Sets

- Let $F$ be the set of suppliers (facilities):  
  $F = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$

- Let $S$ be the set of stores (customers):  
  $S = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$

##### Parameters

- Demand for each store $j \in S$:
  - $\text{Customer}_1$ (CLARINDA): $d_1 = 2397$
  - $\text{Customer}_2$ (FORT MADISON): $d_2 = 1889$
  - $\text{Customer}_3$ (SIOUX CITY): $d_3 = 2518$
  - $\text{Customer}_4$ (TOLEDO): $d_4 = 3218$
  - $\text{Customer}_5$ (BANCROFT): $d_5 = 1813$

- Fixed cost for each supplier $i \in F$:
  - MOUNT AYR: $f_{\text{MOUNT AYR}} = 96.58$
  - WAUKEE: $f_{\text{WAUKEE}} = 94.06$
  - WAVERLY: $f_{\text{WAVERLY}} = 94.37$
  - PELLA: $f_{\text{PELLA}} = 82.88$
  - DES MOINES: $f_{\text{DES MOINES}} = 94.96$

- Transportation cost per unit from supplier $i$ to store $j$ ($c_{ij}$):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA         | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

- Total demand $M = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in F$ to store $j \in S$ (continuous)
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise

##### Objective Function

\[
\min \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij} + \sum_{i \in F} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in S$,
   \[
   \sum_{i \in F} x_{ij} = d_j
   \]

2. **Supplier activation:**  
   For each supplier $i \in F$,
   \[
   \sum_{j \in S} x_{ij} \leq M y_i
   \]
   where $M = 11835$ is the total demand (a valid upper bound since there are no supplier capacity limits).

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in F,\, j \in S
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in F
   \]

##### Parameter Tables

**Demand:**

| Store         | Demand ($d_j$) |
|---------------|---------------|
| CLARINDA      | 2397          |
| FORT MADISON  | 1889          |
| SIOUX CITY    | 2518          |
| TOLEDO        | 3218          |
| BANCROFT      | 1813          |

**Fixed Costs:**

| Supplier      | Fixed Cost ($f_i$) |
|---------------|-------------------|
| MOUNT AYR     | 96.58             |
| WAUKEE        | 94.06             |
| WAVERLY       | 94.37             |
| PELLA         | 82.88             |
| DES MOINES    | 94.96             |

**Transportation Costs ($c_{ij}$):**

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA         | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

##### Summary

Sets, parameters, and all coefficients are as above. The model determines which suppliers to activate and how much each should ship to each store to minimize total cost, subject to demand satisfaction and activation logic.