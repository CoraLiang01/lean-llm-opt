##### Decision Variables

$x_{ij} \geq 0$: Quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Store demand:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j = 11835$.
3. **Domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Sets and Parameters

- Suppliers $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- Stores $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$

###### Demand per store

| Store         | Demand ($d_j$) |
|---------------|---------------|
| Customer_1    | 2397          |
| Customer_2    | 1889          |
| Customer_3    | 2518          |
| Customer_4    | 3218          |
| Customer_5    | 1813          |

(Assuming mapping: Customer_1 = CLARINDA, Customer_2 = FORT MADISON, Customer_3 = SIOUX CITY, Customer_4 = TOLEDO, Customer_5 = BANCROFT)

###### Fixed cost per supplier

| Supplier      | Fixed Cost ($f_i$) |
|---------------|--------------------|
| MOUNT AYR     | 96.58              |
| WAUKEE        | 94.06              |
| WAVERLY       | 94.37              |
| PELLA         | 82.88              |
| DES MOINES    | 94.96              |

###### Transportation cost matrix $c_{ij}$

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

##### Parameter summary

- $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$
- $d = [2397, 1889, 2518, 3218, 1813]$
- $f = [96.58, 94.06, 94.37, 82.88, 94.96]$
- $c =$
  - MOUNT AYR: [694.68, 17.48, 20.07, 199.02, 1685.53]
  - WAUKEE: [15.13, 1.5, 1.43, 27.88, 90.69]
  - WAVERLY: [2.34, 349.34, 246.6, 41.3, 78.73]
  - PELLA: [1181.6, 1458.53, 1646.36, 1924.55, 38.93]
  - DES MOINES: [1030.8, 43.48, 932.43, 55.39, 103.84]
- $M = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

##### Complete Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

where all parameters are as listed above.