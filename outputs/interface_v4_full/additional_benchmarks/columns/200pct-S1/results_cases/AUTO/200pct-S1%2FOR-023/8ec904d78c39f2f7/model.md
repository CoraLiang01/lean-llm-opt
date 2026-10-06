##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i \in I$ to store (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise.

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (suppliers/facilities)
- $J = \{$CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT$\}$ (stores/customers)

- Fixed costs $f_i$:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Demands $d_j$ (from demand.csv, mapped to stores as per identifiers):
  - $d_{\text{CLARINDA}} = 2397$
  - $d_{\text{FORT MADISON}} = 1889$
  - $d_{\text{SIOUX CITY}} = 2518$
  - $d_{\text{TOLEDO}} = 3218$
  - $d_{\text{BANCROFT}} = 1813$

- Transportation costs $c_{ij}$ (facility $i$ to customer $j$):

|                | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|----------|--------------|------------|--------|----------|
| MOUNT AYR      | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE         | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY        | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA          | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES     | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

- Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (sufficiently large upper bound for linking constraints).

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

2. **Supplier activation (linking):**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

**All parameters, vectors, and matrices are explicitly included as retrieved.**