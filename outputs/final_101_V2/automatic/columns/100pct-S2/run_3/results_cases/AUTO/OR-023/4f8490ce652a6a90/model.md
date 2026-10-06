##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i$ to store (customer) $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise (binary).

##### Parameters

- Let $I$ be the set of suppliers (facilities):  
  $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- Let $J$ be the set of stores (customers):  
  $J = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$

- Fixed costs for each supplier $i$:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Demand for each store $j$:
  - $d_{\text{CLARINDA}} = 2397$
  - $d_{\text{FORT MADISON}} = 1889$
  - $d_{\text{SIOUX CITY}} = 2518$
  - $d_{\text{TOLEDO}} = 3218$
  - $d_{\text{BANCROFT}} = 1813$

- Transportation cost per unit from supplier $i$ to store $j$ ($c_{ij}$):

| Supplier $\downarrow$ \ Store $\rightarrow$ | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------------------------------------|----------|--------------|------------|--------|----------|
| MOUNT AYR                                   | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE                                      | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY                                     | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA                                       | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES                                  | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

##### Objective Function

\[
\min \left( \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \right)
\]

##### Constraints

1. **Demand satisfaction:**  
   Each store must receive exactly its demand:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation:**  
   No shipments from inactive suppliers. For each supplier, total shipments cannot exceed a large upper bound $M$ times $y_i$:
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \qquad y_i \in \{0,1\}
   \]

---

###### All parameters, vectors, and matrices are as retrieved and preserved from the original data. No simplification or abbreviation has been performed. The model is ready for direct implementation.