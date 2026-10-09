Let:
- $F$ be the set of suppliers: $\{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- $S$ be the set of customers (stores): $\{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$
- $D$ be the set of demand points: $\{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$

Parameters:
- $f_i$: fixed cost for opening supplier $i$
  - MOUNT AYR: $96.58$
  - WAUKEE: $94.06$
  - WAVERLY: $94.37$
  - PELLA: $82.88$
  - DES MOINES: $94.96$
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$ (see table below)
- $d_j$: demand at customer $j$
  - Customer_1: $2397$
  - Customer_2: $1889$
  - Customer_3: $2518$
  - Customer_4: $3218$
  - Customer_5: $1813$

Transportation cost table ($c_{ij}$):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA         | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

Decision variables:
- $y_i \in \{0,1\}$: $1$ if supplier $i$ is open, $0$ otherwise
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to store $j$

Objective:
\[
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij}
\]

Subject to:

1. **Demand satisfaction** (for each store $j$):
   \[
   \sum_{i \in F} x_{ij} \geq d_j, \quad \forall j \in S
   \]
   (Assign: Customer_1 $\to$ CLARINDA, Customer_2 $\to$ FORT MADISON, Customer_3 $\to$ SIOUX CITY, Customer_4 $\to$ TOLEDO, Customer_5 $\to$ BANCROFT)

   So:
   \[
   \sum_{i \in F} x_{i,\text{CLARINDA}} \geq 2397
   \]
   \[
   \sum_{i \in F} x_{i,\text{FORT MADISON}} \geq 1889
   \]
   \[
   \sum_{i \in F} x_{i,\text{SIOUX CITY}} \geq 2518
   \]
   \[
   \sum_{i \in F} x_{i,\text{TOLEDO}} \geq 3218
   \]
   \[
   \sum_{i \in F} x_{i,\text{BANCROFT}} \geq 1813
   \]

2. **Supplier activation** (no supply from closed suppliers):
   \[
   x_{ij} \leq M \cdot y_i, \quad \forall i \in F, \forall j \in S
   \]
   where $M$ is a sufficiently large constant (e.g., $M = \sum_j d_j$).

3. **Variable domains**:
   \[
   y_i \in \{0,1\}, \quad \forall i \in F
   \]
   \[
   x_{ij} \geq 0, \quad \forall i \in F, \forall j \in S
   \]

All parameters and identifiers are as retrieved and preserved in source order.

**Data used:**

- Fixed costs:
  - MOUNT AYR: $96.58$
  - WAUKEE: $94.06$
  - WAVERLY: $94.37$
  - PELLA: $82.88$
  - DES MOINES: $94.96$

- Transportation costs (see table above).

- Demands:
  - CLARINDA: $2397$
  - FORT MADISON: $1889$
  - SIOUX CITY: $2518$
  - TOLEDO: $3218$
  - BANCROFT: $1813$