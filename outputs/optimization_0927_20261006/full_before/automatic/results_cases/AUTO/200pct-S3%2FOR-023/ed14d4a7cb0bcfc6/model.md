##### Decision Variables

Let:
- $x_{ij} \geq 0$: quantity of liquor product shipped from supplier (facility) $i$ to store (customer) $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (set of suppliers/facilities)
- $J = \{$Customer_1, Customer_2, Customer_3, Customer_4, Customer_5$\}$ (set of stores/customers)
- $f_i$: fixed cost for opening supplier $i$ (from fixed_cost.csv, current period)
- $d_j$: demand at store $j$ (from demand.csv, current period)
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv, current period)

###### Fixed Costs ($f_i$):

| Supplier      | Fixed Cost ($f_i$) |
|---------------|--------------------|
| MOUNT AYR     | 96.58              |
| WAUKEE        | 94.06              |
| WAVERLY       | 94.37              |
| PELLA         | 82.88              |
| DES MOINES    | 94.96              |

###### Demands ($d_j$):

| Store        | Demand ($d_j$) |
|--------------|----------------|
| Customer_1   | 2397           |
| Customer_2   | 1889           |
| Customer_3   | 2518           |
| Customer_4   | 3218           |
| Customer_5   | 1813           |

###### Transportation Costs ($c_{ij}$):

| Supplier \ Store | Customer_1 (CLARINDA) | Customer_2 (FORT MADISON) | Customer_3 (SIOUX CITY) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
|------------------|-----------------------|---------------------------|-------------------------|---------------------|-----------------------|
| MOUNT AYR        | 694.68                | 17.48                     | 20.07                   | 199.02              | 1685.53               |
| WAUKEE           | 15.13                 | 1.5                       | 1.43                    | 27.88               | 90.69                 |
| WAVERLY          | 2.34                  | 349.34                    | 246.60                  | 41.30               | 78.73                 |
| PELLA            | 1181.60               | 1458.53                   | 1646.36                 | 1924.55             | 38.93                 |
| DES MOINES       | 1030.80               | 43.48                     | 932.43                  | 55.39               | 103.84                |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand Satisfaction:**  
   Each store's demand must be fully met:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier Activation:**  
   No shipments from inactive suppliers:
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$.

3. **Variable Domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq 11835 \cdot y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

Where:

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$
- $J = \{$Customer_1, Customer_2, Customer_3, Customer_4, Customer_5$\}$
- $f_i$, $d_j$, $c_{ij}$ as specified above.

All parameters, vectors, and matrices are as retrieved from the CSV data, with no simplification or abbreviation.