##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i$ to store $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Parameters

- Let $I$ be the set of suppliers (indexed by $i$):  
  $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- Let $J$ be the set of stores (indexed by $j$):  
  $J = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$
- Let $d_j$ be the demand at store $j$ (from demand.csv, mapped below).
- Let $f_i$ be the fixed cost for opening supplier $i$ (from fixed_cost.csv).
- Let $c_{ij}$ be the transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv).
- Let $M = \sum_{j \in J} d_j$ (an upper bound on total shipments from any supplier).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation:**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
3. **Variable domains:**
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

---

#### Data Mapping

- **Suppliers $I$ (from fixed_cost.csv, column "Unnamed: 0"):**
  - MOUNT AYR
  - WAUKEE
  - WAVERLY
  - PELLA
  - DES MOINES

- **Stores $J$ (from transportation_costs.csv, columns):**
  - CLARINDA
  - FORT MADISON
  - SIOUX CITY
  - TOLEDO
  - BANCROFT

- **Demands $d_j$ (from demand.csv, column "demand", mapped in order):**
  - $d_{\text{CLARINDA}} = 2397$
  - $d_{\text{FORT MADISON}} = 1889$
  - $d_{\text{SIOUX CITY}} = 2518$
  - $d_{\text{TOLEDO}} = 3218$
  - $d_{\text{BANCROFT}} = 1813$

- **Fixed costs $f_i$ (from fixed_cost.csv, column "fixed_costs"):**
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- **Transportation costs $c_{ij}$ (from transportation_costs.csv):**

| Supplier $\backslash$ Store | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------------------|----------|--------------|------------|--------|----------|
| MOUNT AYR                  | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE                     | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY                    | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA                      | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES                 | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

- **Big-M parameter:**
  - $M = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

---

**Source-Column Data Mapping:**

- demand.csv: Customer $\rightarrow$ store $j$ (in order), demand $\rightarrow d_j$
- fixed_cost.csv: Unnamed: 0 $\rightarrow$ supplier $i$, fixed_costs $\rightarrow f_i$
- transportation_costs.csv: Unnamed: 0 $\rightarrow$ supplier $i$, columns $\rightarrow$ store $j$, values $\rightarrow c_{ij}$

---

**Summary:**  
This is a capacitated facility location model with binary supplier activation and continuous shipment variables, using all data as mapped above. The objective is to minimize total fixed and transportation costs while meeting all store demands.