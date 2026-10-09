##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (operational), 0 otherwise (binary).

##### Sets

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (Suppliers)
- $J = \{$Customer_1, Customer_2, Customer_3, Customer_4, Customer_5$\}$ (Stores)

##### Parameters

- Demand for each store:
  - $d_{Customer_1} = 2397$
  - $d_{Customer_2} = 1889$
  - $d_{Customer_3} = 2518$
  - $d_{Customer_4} = 3218$
  - $d_{Customer_5} = 1813$

- Fixed cost for each supplier:
  - $f_{MOUNT AYR} = 96.58$
  - $f_{WAUKEE} = 94.06$
  - $f_{WAVERLY} = 94.37$
  - $f_{PELLA} = 82.88$
  - $f_{DES MOINES} = 94.96$

- Transportation cost per unit from each supplier to each store (matrix $c_{ij}$):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA         | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

**Note:** The mapping from store names (Customer_1, ..., Customer_5) to city names (CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT) is assumed as follows for the model, based on the order of appearance:

- Customer_1 $\rightarrow$ CLARINDA
- Customer_2 $\rightarrow$ FORT MADISON
- Customer_3 $\rightarrow$ SIOUX CITY
- Customer_4 $\rightarrow$ TOLEDO
- Customer_5 $\rightarrow$ BANCROFT

So, $c_{ij}$ is:

|               | Customer_1 (CLARINDA) | Customer_2 (FORT MADISON) | Customer_3 (SIOUX CITY) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
|---------------|-----------------------|---------------------------|-------------------------|---------------------|-----------------------|
| MOUNT AYR     | 694.68                | 17.48                    | 20.07                  | 199.02              | 1685.53               |
| WAUKEE        | 15.13                 | 1.50                     | 1.43                   | 27.88               | 90.69                 |
| WAVERLY       | 2.34                  | 349.34                   | 246.60                 | 41.30               | 78.73                 |
| PELLA         | 1181.60               | 1458.53                  | 1646.36                | 1924.55             | 38.93                 |
| DES MOINES    | 1030.80               | 43.48                    | 932.43                 | 55.39               | 103.84                |

- $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:** Each store's demand must be met.
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

##### Summary of Parameters

- Suppliers: $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$
- Stores: $J = \{$Customer_1, Customer_2, Customer_3, Customer_4, Customer_5$\}$
- Demands: $d_{Customer_1} = 2397$, $d_{Customer_2} = 1889$, $d_{Customer_3} = 2518$, $d_{Customer_4} = 3218$, $d_{Customer_5} = 1813$
- Fixed costs: $f_{MOUNT AYR} = 96.58$, $f_{WAUKEE} = 94.06$, $f_{WAVERLY} = 94.37$, $f_{PELLA} = 82.88$, $f_{DES MOINES} = 94.96$
- Transportation costs $c_{ij}$ as in the table above
- $M = 11835$