##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i$ to store (customer) $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise (binary).

##### Sets

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (suppliers)
- $J = \{$Customer\_1, Customer\_2, Customer\_3, Customer\_4, Customer\_5$\}$ (stores)

##### Parameters

- $f_i$: Fixed cost for opening supplier $i$ (current period):

  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- $d_j$: Demand at store $j$ (current period):

  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$

- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (current period):

  | Supplier $\downarrow$ \ Store $\rightarrow$ | Customer_1 (SIOUX CITY) | Customer_2 (CLARINDA) | Customer_3 (FORT MADISON) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
  |---------------------------------------------|-------------------------|-----------------------|---------------------------|---------------------|-----------------------|
  | MOUNT AYR                                   | 20.07                   | 694.68                | 17.48                     | 199.02              | 1685.53               |
  | WAUKEE                                      | 1.43                    | 15.13                 | 1.5                       | 27.88               | 90.69                 |
  | WAVERLY                                     | 246.6                   | 2.34                  | 349.34                    | 41.3                | 78.73                 |
  | PELLA                                       | 1646.36                 | 1181.6                | 1458.53                   | 1924.55             | 38.93                 |
  | DES MOINES                                  | 932.43                  | 1030.8                | 43.48                     | 55.39               | 103.84                |

- $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (sufficiently large upper bound for each supplier's total shipment)

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

###### Retrieved Information

- **Suppliers (Facilities):** MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- **Stores (Customers):** Customer_1 (SIOUX CITY), Customer_2 (CLARINDA), Customer_3 (FORT MADISON), Customer_4 (TOLEDO), Customer_5 (BANCROFT)
- **Fixed costs (current period):**
  - MOUNT AYR: 96.58
  - WAUKEE: 94.06
  - WAVERLY: 94.37
  - PELLA: 82.88
  - DES MOINES: 94.96
- **Demand (current period):**
  - Customer_1: 2397
  - Customer_2: 1889
  - Customer_3: 2518
  - Customer_4: 3218
  - Customer_5: 1813
- **Transportation cost matrix (current period):**

  | Supplier \ Store | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
  |------------------|------------|------------|------------|------------|------------|
  | MOUNT AYR        | 20.07      | 694.68     | 17.48      | 199.02     | 1685.53    |
  | WAUKEE           | 1.43       | 15.13      | 1.5        | 27.88      | 90.69      |
  | WAVERLY          | 246.6      | 2.34       | 349.34     | 41.3       | 78.73      |
  | PELLA            | 1646.36    | 1181.6     | 1458.53    | 1924.55    | 38.93      |
  | DES MOINES       | 932.43     | 1030.8     | 43.48      | 55.39      | 103.84     |

- **Big-M value:** $M = 11835$

---

This model determines which suppliers to open and how much each should ship to each store to minimize total cost, while meeting all store demands and only allowing shipments from open suppliers. All parameters and identifiers are preserved exactly as in the provided data.