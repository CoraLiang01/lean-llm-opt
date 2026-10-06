##### Decision Variables

$x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i$ to store (customer) $j$, for all $i \in I$, $j \in J$ (continuous).

$y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

---

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (set of suppliers/facilities)
- $J = \{$Customer\_1, Customer\_2, Customer\_3, Customer\_4, Customer\_5$\}$ (set of stores/customers)
- $d_j$: Demand at customer $j$ (from demand.csv)
- $f_i$: Fixed cost for opening supplier $i$ (from fixed_cost.csv)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv)
- $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (sufficiently large upper bound for linking constraints)

---

###### Demand Data

| Customer    | Demand |
|-------------|--------|
| Customer_1  | 2397   |
| Customer_2  | 1889   |
| Customer_3  | 2518   |
| Customer_4  | 3218   |
| Customer_5  | 1813   |

###### Fixed Cost Data

| Facility      | Fixed_Cost |
|---------------|------------|
| MOUNT AYR     | 96.58      |
| WAUKEE        | 94.06      |
| WAVERLY       | 94.37      |
| PELLA         | 82.88      |
| DES MOINES    | 94.96      |

###### Transportation Cost Data

| Facility      | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|---------------|------------|------------|------------|------------|------------|
| MOUNT AYR     | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |
| WAUKEE        | 15.13      | 1.50       | 1.43       | 27.88      | 90.69      |
| WAVERLY       | 2.34       | 349.34     | 246.60     | 41.30      | 78.73      |
| PELLA         | 1181.60    | 1458.53    | 1646.36    | 1924.55    | 38.93      |
| DES MOINES    | 1030.80    | 43.48      | 932.43     | 55.39      | 103.84     |

---

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

Where:
- $c_{ij}$ is the transportation cost per unit from facility $i$ to customer $j$ (see table above).
- $f_i$ is the fixed cost for opening facility $i$ (see table above).

---

##### Constraints

1. **Demand Satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
   (Each store's demand must be fully met.)

2. **Facility Activation:**  
   For each facility $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   (No shipments from a facility unless it is open; $M = 11835$.)

3. **Variable Domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

Where all parameters and sets are as defined above, with explicit values from the CSV data.