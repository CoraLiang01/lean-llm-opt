##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i \in I$ to store (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (Suppliers/facilities)
- $J = \{$Customer_1, Customer_2, Customer_3, Customer_4, Customer_5$\}$ (Stores/customers)
- $f_i$: Fixed cost for opening supplier $i$:

| Supplier      | Fixed Cost ($f_i$) |
|---------------|-------------------|
| MOUNT AYR     | 96.58             |
| WAUKEE        | 94.06             |
| WAVERLY       | 94.37             |
| PELLA         | 82.88             |
| DES MOINES    | 94.96             |

- $d_j$: Demand at store $j$:

| Store        | Demand ($d_j$) |
|--------------|---------------|
| Customer_1   | 2397          |
| Customer_2   | 1889          |
| Customer_3   | 2518          |
| Customer_4   | 3218          |
| Customer_5   | 1813          |

- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$:

| Supplier    | Customer_1 (CLARINDA) | Customer_2 (FORT MADISON) | Customer_3 (SIOUX CITY) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
|-------------|----------------------|---------------------------|-------------------------|---------------------|-----------------------|
| MOUNT AYR   | 694.68               | 17.48                     | 20.07                   | 199.02              | 1685.53               |
| WAUKEE      | 15.13                | 1.50                      | 1.43                    | 27.88               | 90.69                 |
| WAVERLY     | 2.34                 | 349.34                    | 246.60                  | 41.30               | 78.73                 |
| PELLA       | 1181.60              | 1458.53                   | 1646.36                 | 1924.55             | 38.93                 |
| DES MOINES  | 1030.80              | 43.48                     | 932.43                  | 55.39               | 103.84                |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:** Each store's demand must be met exactly.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation:** No shipments from inactive suppliers.
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$.

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
- Customer_1
- Customer_2
- Customer_3
- Customer_4
- Customer_5

**Demands ($d_j$):**

| Customer    | Demand |
|-------------|--------|
| Customer_1  | 2397   |
| Customer_2  | 1889   |
| Customer_3  | 2518   |
| Customer_4  | 3218   |
| Customer_5  | 1813   |

**Fixed Costs ($f_i$):**

| Supplier      | Fixed Cost |
|---------------|------------|
| MOUNT AYR     | 96.58      |
| WAUKEE        | 94.06      |
| WAVERLY       | 94.37      |
| PELLA         | 82.88      |
| DES MOINES    | 94.96      |

**Transportation Costs ($c_{ij}$):**

| Supplier    | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|-------------|------------|------------|------------|------------|------------|
| MOUNT AYR   | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |
| WAUKEE      | 15.13      | 1.50       | 1.43       | 27.88      | 90.69      |
| WAVERLY     | 2.34       | 349.34     | 246.60     | 41.30      | 78.73      |
| PELLA       | 1181.60    | 1458.53    | 1646.36    | 1924.55    | 38.93      |
| DES MOINES  | 1030.80    | 43.48      | 932.43     | 55.39      | 103.84     |

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

Where all parameters are as listed above.