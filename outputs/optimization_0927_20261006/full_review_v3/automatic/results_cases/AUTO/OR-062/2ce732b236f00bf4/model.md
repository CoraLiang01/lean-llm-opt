##### Decision Variables

$x_{ij} \geq 0$: quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. Store demand: $\sum_{i\in I} x_{ij} = d_j,\quad \forall j\in J$
2. Supplier activation: $\sum_{j\in J} x_{ij} \leq M y_i,\quad \forall i\in I$
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

Where $M = \sum_{j\in J} d_j = 11835$ is a valid upper bound on total shipments from any supplier.

---

#### Parameters

- Suppliers ($I$):  
  $\{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$

- Stores ($J$):  
  $\{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$

- Store demands ($d_j$):  
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$

- Supplier fixed costs ($f_i$):  
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Transportation costs ($c_{ij}$):  
  | Supplier $\downarrow$ \ Store $\rightarrow$ | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
  |:----------------------|:---------:|:------------:|:----------:|:------:|:--------:|
  | MOUNT AYR             | 694.68    | 17.48        | 20.07      | 199.02 | 1685.53  |
  | WAUKEE                | 15.13     | 1.50         | 1.43       | 27.88  | 90.69    |
  | WAVERLY               | 2.34      | 349.34       | 246.60     | 41.30  | 78.73    |
  | PELLA                 | 1181.60   | 1458.53      | 1646.36    | 1924.55| 38.93    |
  | DES MOINES            | 1030.80   | 43.48        | 932.43     | 55.39  | 103.84   |

---

###### Retrieved Information

{
  "suppliers": [
    "MOUNT AYR",
    "WAUKEE",
    "WAVERLY",
    "PELLA",
    "DES MOINES"
  ],
  "stores": [
    "CLARINDA",
    "FORT MADISON",
    "SIOUX CITY",
    "TOLEDO",
    "BANCROFT"
  ],
  "demand": {
    "Customer_1": 2397,
    "Customer_2": 1889,
    "Customer_3": 2518,
    "Customer_4": 3218,
    "Customer_5": 1813
  },
  "fixed_cost": {
    "MOUNT AYR": 96.58,
    "WAUKEE": 94.06,
    "WAVERLY": 94.37,
    "PELLA": 82.88,
    "DES MOINES": 94.96
  },
  "cost": {
    "MOUNT AYR": {
      "CLARINDA": 694.68,
      "FORT MADISON": 17.48,
      "SIOUX CITY": 20.07,
      "TOLEDO": 199.02,
      "BANCROFT": 1685.53
    },
    "WAUKEE": {
      "CLARINDA": 15.13,
      "FORT MADISON": 1.50,
      "SIOUX CITY": 1.43,
      "TOLEDO": 27.88,
      "BANCROFT": 90.69
    },
    "WAVERLY": {
      "CLARINDA": 2.34,
      "FORT MADISON": 349.34,
      "SIOUX CITY": 246.60,
      "TOLEDO": 41.30,
      "BANCROFT": 78.73
    },
    "PELLA": {
      "CLARINDA": 1181.60,
      "FORT MADISON": 1458.53,
      "SIOUX CITY": 1646.36,
      "TOLEDO": 1924.55,
      "BANCROFT": 38.93
    },
    "DES MOINES": {
      "CLARINDA": 1030.80,
      "FORT MADISON": 43.48,
      "SIOUX CITY": 932.43,
      "TOLEDO": 55.39,
      "BANCROFT": 103.84
    }
  },
  "M": 11835
}