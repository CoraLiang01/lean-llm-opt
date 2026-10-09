##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i \in I$ to store (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Sets

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (suppliers/facilities)
- $J = \{$CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT$\}$ (stores/customers)

##### Parameters

- Demands (from demand.csv):

  | Customer    | Demand |
  |-------------|--------|
  | Customer_1  | 2397   |
  | Customer_2  | 1889   |
  | Customer_3  | 2518   |
  | Customer_4  | 3218   |
  | Customer_5  | 1813   |

  Assigning:  
  $d_1 = 2397$ (Customer_1),  
  $d_2 = 1889$ (Customer_2),  
  $d_3 = 2518$ (Customer_3),  
  $d_4 = 3218$ (Customer_4),  
  $d_5 = 1813$ (Customer_5)$

- Fixed costs (from fixed_cost.csv):

  | Facility      | Fixed Cost |
  |---------------|------------|
  | MOUNT AYR     | 96.58      |
  | WAUKEE        | 94.06      |
  | WAVERLY       | 94.37      |
  | PELLA         | 82.88      |
  | DES MOINES    | 94.96      |

  $f_{\text{MOUNT AYR}} = 96.58$  
  $f_{\text{WAUKEE}} = 94.06$  
  $f_{\text{WAVERLY}} = 94.37$  
  $f_{\text{PELLA}} = 82.88$  
  $f_{\text{DES MOINES}} = 94.96$

- Transportation costs (from transportation_costs.csv):

  | Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
  |---------------|----------|--------------|------------|--------|----------|
  | MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
  | WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
  | WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
  | PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
  | DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

  Let $c_{ij}$ denote the transportation cost per unit from supplier $i$ to store $j$.

##### Mathematical Model

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

**Subject to:**

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
   where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$.

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Parameters (explicit):

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$
- $J = \{$CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT$\}$
- Demands:
  - $d_{\text{CLARINDA}} = 2397$
  - $d_{\text{FORT MADISON}} = 1889$
  - $d_{\text{SIOUX CITY}} = 2518$
  - $d_{\text{TOLEDO}} = 3218$
  - $d_{\text{BANCROFT}} = 1813$
- Fixed costs:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$
- Transportation costs $c_{ij}$:

  | $c_{ij}$      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
  |---------------|----------|--------------|------------|--------|----------|
  | MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
  | WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
  | WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
  | PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
  | DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

- $M = 11835$

---

**Summary:**  
This model determines which suppliers to activate and how much each should ship to each store, so that all store demands are met at minimum total cost (fixed + transportation), with suppliers only shipping if activated. All parameters and sets are explicitly listed as retrieved from the CSV files.