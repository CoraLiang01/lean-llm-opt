##### Decision Variables

Let:
- $x_{ij} \geq 0$: quantity of goods shipped from supplier (facility) $i$ to store (customer) $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (set of suppliers)
- $J = \{$Customer_1, Customer_2, Customer_3, Customer_4, Customer_5$\}$ (set of stores)
- $f_i$: fixed cost for opening supplier $i$ (from fixed_cost.csv, "FixedCost (Current)")
- $d_j$: demand at store $j$ (from demand.csv, "Demand (Current)")
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv, "Cost (Current)")

##### Data

- Fixed costs:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Demands:
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$

- Transportation costs matrix $c_{ij}$ (supplier rows, store columns):

|                | Customer_1 (CLARINDA) | Customer_2 (FORT MADISON) | Customer_3 (SIOUX CITY) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
|----------------|-----------------------|---------------------------|-------------------------|---------------------|-----------------------|
| MOUNT AYR      | 694.68                | 17.48                     | 20.07                   | 199.02              | 1685.53               |
| WAUKEE         | 15.13                 | 1.5                       | 1.43                    | 27.88               | 90.69                 |
| WAVERLY        | 2.34                  | 349.34                    | 246.60                  | 41.30               | 78.73                 |
| PELLA          | 1181.60               | 1458.53                   | 1646.36                 | 1924.55             | 38.93                 |
| DES MOINES     | 1030.80               | 43.48                     | 932.43                  | 55.39               | 103.84                |

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
   \sum_{j \in J} x_{ij} \leq M \cdot y_i
   \]
   where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$.

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
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
- $f_i$ and $c_{ij}$ as specified above
- $d_j$ as specified above
- $M = 11835$