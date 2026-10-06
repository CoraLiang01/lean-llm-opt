##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i \in I$ to store (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$ (set of suppliers/facilities)
- $J = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$ (set of stores/customers)
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
- Transportation costs $c_{ij}$ (per unit from facility $i$ to customer $j$):

| Facility \ Customer   | Customer_1 (CLARINDA) | Customer_2 (FORT MADISON) | Customer_3 (SIOUX CITY) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
|----------------------|-----------------------|---------------------------|-------------------------|---------------------|-----------------------|
| MOUNT AYR            | 694.68                | 17.48                     | 20.07                   | 199.02              | 1685.53               |
| WAUKEE               | 15.13                 | 1.50                      | 1.43                    | 27.88               | 90.69                 |
| WAVERLY              | 2.34                  | 349.34                    | 246.60                  | 41.30               | 78.73                 |
| PELLA                | 1181.60               | 1458.53                   | 1646.36                 | 1924.55             | 38.93                 |
| DES MOINES           | 1030.80               | 43.48                     | 932.43                  | 55.39               | 103.84                |

##### Objective Function

\[
\min \left( \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \right)
\]

##### Constraints

1. **Demand satisfaction:** Each store's demand must be met exactly.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation:** No shipments from a supplier unless it is open.
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (a valid upper bound since there are no explicit supplier capacities).

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq 11835\, y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

##### Data Used

- Facilities (Suppliers): MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- Fixed Costs: 96.58, 94.06, 94.37, 82.88, 94.96 (in order above)
- Customers (Stores): Customer_1, Customer_2, Customer_3, Customer_4, Customer_5
- Demands: 2397, 1889, 2518, 3218, 1813 (in order above)
- Transportation Cost Matrix (see table above)
- $M = 11835$ (sum of all demands)

All identifiers, coefficients, and matrix axes are preserved as in the original data.