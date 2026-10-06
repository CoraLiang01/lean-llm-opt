##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i$ to store (customer) $j$.
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (operational), 0 otherwise.

##### Sets

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (supplier/facility IDs)
- $J = \{$CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT$\}$ (store/customer IDs)

##### Parameters

- Demand for each store (current period):

  $d_j$:
  - $d_{\text{CLARINDA}} = 2397$
  - $d_{\text{FORT MADISON}} = 1889$
  - $d_{\text{SIOUX CITY}} = 2518$
  - $d_{\text{TOLEDO}} = 3218$
  - $d_{\text{BANCROFT}} = 1813$

- Fixed cost for each supplier (current period):

  $f_i$:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Transportation cost per unit from supplier $i$ to store $j$ (current period):

  $c_{ij}$:

  |                | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
  |----------------|----------|--------------|------------|--------|----------|
  | MOUNT AYR      | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
  | WAUKEE         | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
  | WAVERLY        | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
  | PELLA          | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
  | DES MOINES     | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

- $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:** Each store must receive exactly its demand.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation:** No shipments from inactive suppliers.
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Complete Model Data

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$
- $J = \{$CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT$\}$
- $d_j$ as above
- $f_i$ as above
- $c_{ij}$ as above
- $M = 11835$

This model determines which suppliers to activate and how much each should ship to each store to meet all demands at minimum total cost.