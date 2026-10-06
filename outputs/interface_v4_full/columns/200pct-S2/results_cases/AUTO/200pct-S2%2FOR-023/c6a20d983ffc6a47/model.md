##### Decision Variables

$x_{ij} \geq 0$: Quantity of liquor product shipped from facility (supplier) $i$ to customer (store) $j$, for all $i \in I$, $j \in J$ (continuous).

$y_i \in \{0,1\}$: 1 if facility (supplier) $i$ is operational (open), 0 otherwise, for all $i \in I$ (binary).

---

##### Parameters

- $I = \{1,2,3,4,5\}$: Set of facilities (suppliers), corresponding to:
  - 1: MOUNT AYR
  - 2: WAUKEE
  - 3: WAVERLY
  - 4: PELLA
  - 5: DES MOINES

- $J = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$: Set of customers (stores).

- Fixed costs $f_i$ for each facility $i$:
  - $f_1 = 96.58$ (MOUNT AYR)
  - $f_2 = 94.06$ (WAUKEE)
  - $f_3 = 94.37$ (WAVERLY)
  - $f_4 = 82.88$ (PELLA)
  - $f_5 = 94.96$ (DES MOINES)

- Demands $d_j$ for each customer $j$:
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$

- Transportation costs $c_{ij}$ (cost per unit from facility $i$ to customer $j$):

| $c_{ij}$         | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|------------------|------------|------------|------------|------------|------------|
| MOUNT AYR (1)    | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |
| WAUKEE (2)       | 15.13      | 1.50       | 1.43       | 27.88      | 90.69      |
| WAVERLY (3)      | 2.34       | 349.34     | 246.60     | 41.30      | 78.73      |
| PELLA (4)        | 1181.60    | 1458.53    | 1646.36    | 1924.55    | 38.93      |
| DES MOINES (5)   | 1030.80    | 43.48      | 932.43     | 55.39      | 103.84     |

---

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

That is, minimize the total transportation cost plus the total fixed cost of opening facilities.

---

##### Constraints

1. **Demand satisfaction:** Each customer’s demand must be fully met.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Facility activation:** No shipments can be made from a facility unless it is open.
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   Where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (a valid upper bound, since there are no explicit facility capacities).

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

---

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq 11835\, y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

Where:

- $I = \{1,2,3,4,5\}$ (MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES)
- $J = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$
- $f_i$, $d_j$, $c_{ij}$ as specified above
- $M = 11835$

---

###### Retrieved Information

```json
{
  "facilities": [
    {"id": 1, "name": "MOUNT AYR", "fixed_cost": 96.58},
    {"id": 2, "name": "WAUKEE", "fixed_cost": 94.06},
    {"id": 3, "name": "WAVERLY", "fixed_cost": 94.37},
    {"id": 4, "name": "PELLA", "fixed_cost": 82.88},
    {"id": 5, "name": "DES MOINES", "fixed_cost": 94.96}
  ],
  "customers": [
    {"id": "Customer_1", "demand": 2397},
    {"id": "Customer_2", "demand": 1889},
    {"id": "Customer_3", "demand": 2518},
    {"id": "Customer_4", "demand": 3218},
    {"id": "Customer_5", "demand": 1813}
  ],
  "transportation_costs": {
    "MOUNT AYR":    {"Customer_1": 694.68, "Customer_2": 17.48, "Customer_3": 20.07, "Customer_4": 199.02, "Customer_5": 1685.53},
    "WAUKEE":       {"Customer_1": 15.13,  "Customer_2": 1.50,  "Customer_3": 1.43,  "Customer_4": 27.88,  "Customer_5": 90.69},
    "WAVERLY":      {"Customer_1": 2.34,   "Customer_2": 349.34,"Customer_3": 246.60,"Customer_4": 41.30,  "Customer_5": 78.73},
    "PELLA":        {"Customer_1": 1181.60,"Customer_2": 1458.53,"Customer_3": 1646.36,"Customer_4": 1924.55,"Customer_5": 38.93},
    "DES MOINES":   {"Customer_1": 1030.80,"Customer_2": 43.48, "Customer_3": 932.43,"Customer_4": 55.39,  "Customer_5": 103.84}
  },
  "M": 11835
}
```