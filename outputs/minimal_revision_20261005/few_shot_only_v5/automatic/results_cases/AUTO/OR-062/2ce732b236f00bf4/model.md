##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i$ to store $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Parameters

- Let $I$ be the set of suppliers (indexed by $i$), and $J$ the set of stores (indexed by $j$).
- $f_i$: Fixed cost to activate supplier $i$.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$.
- $d_j$: Demand at store $j$.
- $M = \sum_{j \in J} d_j$ (a valid upper bound for total shipments from any supplier, since there are no supplier capacity limits).

##### Sets and Indices

- Suppliers $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- Stores $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand Satisfaction:**  
   For each store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
2. **Supplier Activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
3. **Variable Domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Data Mapping

- **Suppliers ($I$) and Fixed Costs ($f_i$):**  
  - MOUNT AYR: $f_{\text{MOUNT AYR}} = 96.58$
  - WAUKEE: $f_{\text{WAUKEE}} = 94.06$
  - WAVERLY: $f_{\text{WAVERLY}} = 94.37$
  - PELLA: $f_{\text{PELLA}} = 82.88$
  - DES MOINES: $f_{\text{DES MOINES}} = 94.96$

- **Stores ($J$) and Demands ($d_j$):**  
  - CLARINDA: $d_{\text{CLARINDA}} = 2397$
  - FORT MADISON: $d_{\text{FORT MADISON}} = 1889$
  - SIOUX CITY: $d_{\text{SIOUX CITY}} = 2518$
  - TOLEDO: $d_{\text{TOLEDO}} = 3218$
  - BANCROFT: $d_{\text{BANCROFT}} = 1813$

- **Transportation Costs ($c_{ij}$):**  
  | Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
  |---------------|----------|--------------|------------|--------|----------|
  | MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
  | WAUKEE        | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
  | WAVERLY       | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
  | PELLA         | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
  | DES MOINES    | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

- **Big-M Value:**  
  $M = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

##### Source-Column Data Mapping

- Supplier set and fixed costs: `/UFLP6/fixed_cost.csv`, columns `Unnamed: 0`, `fixed_costs`
- Store set and demands: `/UFLP6/demand.csv`, columns `Customer`, `demand`
- Transportation costs: `/UFLP6/transportation_costs.csv`, rows indexed by supplier, columns by store

---

**Summary:**  
This model determines which suppliers to activate and how much each should ship to each store, so that all store demands are met at minimum total cost (fixed + transportation), with no supplier capacity limits. All data is mapped directly from the provided CSV columns.