##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i \in I$ to store (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (set of suppliers/facilities)
- $J = \{$CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT$\}$ (set of stores/customers)
- $f_i$: Fixed cost for opening supplier $i$ (from fixed_cost.csv)
- $d_j$: Demand at store $j$ (from demand.csv)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv)
- $M = \sum_{j \in J} d_j$ (a sufficiently large constant for linking constraints)

##### Data

###### Fixed Costs ($f_i$)

| Supplier (Facility) | Fixed Cost |
|---------------------|------------|
| MOUNT AYR           | 96.58      |
| WAUKEE              | 94.06      |
| WAVERLY             | 94.37      |
| PELLA               | 82.88      |
| DES MOINES          | 94.96      |

###### Store Demands ($d_j$)

| Store (Customer) | Demand |
|------------------|--------|
| CLARINDA         | 2397   |
| FORT MADISON     | 1889   |
| SIOUX CITY       | 2518   |
| TOLEDO           | 3218   |
| BANCROFT         | 1813   |

Total demand $M = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

###### Transportation Costs ($c_{ij}$)

| Supplier \ Store | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|------------------|----------|--------------|------------|--------|----------|
| MOUNT AYR        | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE           | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY          | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA            | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES       | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

##### Mathematical Model

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

**Subject to:**

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation constraint:**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   (where $M = 11835$)

3. **Variable domains:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Parameter Summary

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$
- $J = \{$CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT$\}$
- $f_i$ as above
- $d_j$ as above
- $c_{ij}$ as above
- $M = 11835$

All data is preserved as in the original files. This model determines which suppliers to open and how much each should ship to each store to minimize total cost while meeting all store demands.