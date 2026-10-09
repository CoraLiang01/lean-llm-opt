##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Sets

- Suppliers $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- Stores $J = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$

##### Parameters

- Store demands (from demand.csv):

  | Store         | Demand |
  |---------------|--------|
  | Customer_1    | 2397   |
  | Customer_2    | 1889   |
  | Customer_3    | 2518   |
  | Customer_4    | 3218   |
  | Customer_5    | 1813   |

  (Assuming mapping: Customer_1 → CLARINDA, Customer_2 → FORT MADISON, Customer_3 → SIOUX CITY, Customer_4 → TOLEDO, Customer_5 → BANCROFT)

  $d_j$:
  - $d_{\text{CLARINDA}} = 2397$
  - $d_{\text{FORT MADISON}} = 1889$
  - $d_{\text{SIOUX CITY}} = 2518$
  - $d_{\text{TOLEDO}} = 3218$
  - $d_{\text{BANCROFT}} = 1813$

- Supplier fixed costs (from fixed_cost.csv):

  | Supplier      | Fixed Cost |
  |--------------|------------|
  | MOUNT AYR    | 96.58      |
  | WAUKEE       | 94.06      |
  | WAVERLY      | 94.37      |
  | PELLA        | 82.88      |
  | DES MOINES   | 94.96      |

  $f_i$:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Transportation costs $c_{ij}$ (from transportation_costs.csv):

  | Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
  |---------------|----------|--------------|------------|--------|----------|
  | MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
  | WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
  | WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
  | PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
  | DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

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
   where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Full Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j, && \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq 11835\, y_i, && \forall i \in I \\
& x_{ij} \geq 0,\, y_i \in \{0,1\} && \forall i \in I,\, j \in J
\end{align*}
\]

##### Parameters (explicit):

- $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- $J = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$
- $d_{\text{CLARINDA}} = 2397$, $d_{\text{FORT MADISON}} = 1889$, $d_{\text{SIOUX CITY}} = 2518$, $d_{\text{TOLEDO}} = 3218$, $d_{\text{BANCROFT}} = 1813$
- $f_{\text{MOUNT AYR}} = 96.58$, $f_{\text{WAUKEE}} = 94.06$, $f_{\text{WAVERLY}} = 94.37$, $f_{\text{PELLA}} = 82.88$, $f_{\text{DES MOINES}} = 94.96$
- $c_{ij}$ as in the table above
- $M = 11835$