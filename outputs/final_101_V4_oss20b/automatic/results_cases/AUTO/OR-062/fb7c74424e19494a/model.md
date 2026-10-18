##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij}+\sum_{i\in I}f_i y_i$

##### Constraints

1. Store demand: $\sum_{i\in I}x_{ij}=d_j,\quad \forall j\in J$.  
2. Supplier activation: $\sum_{j\in J}x_{ij}\leq M y_i,\quad \forall i\in I$.  
3. Domains: $x_{ij}\geq0$ continuous; $y_i\in\{0,1\}$.

$I=\{\text{MOUNT AYR},\text{WAUKEE},\text{WAVERLY},\text{PELLA},\text{DES MOINES}\}$,  
$J=\{\text{CLARINDA},\text{FORT MADISON},\text{SIOUX CITY},\text{TOLEDO},\text{BANCROFT}\}$,  
$M=\sum_j d_j=11835$.  

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
    "CLARINDA": 2397,
    "FORT MADISON": 1889,
    "SIOUX CITY": 2518,
    "TOLEDO": 3218,
    "BANCROFT": 1813
  },
  "fixed_cost": {
    "MOUNT AYR": 96.58,
    "WAUKEE": 94.06,
    "WAVERLY": 94.37,
    "PELLA": 82.88,
    "DES MOINES": 94.95999999999999
  },
  "cost": {
    "MOUNT AYR": {
      "CLARINDA": 694.6799999999999,
      "FORT MADISON": 17.48,
      "SIOUX CITY": 20.07,
      "TOLEDO": 199.02,
      "BANCROFT": 1685.53
    },
    "WAUKEE": {
      "CLARINDA": 15.13,
      "FORT MADISON": 1.5,
      "SIOUX CITY": 1.43,
      "TOLEDO": 27.88,
      "BANCROFT": 90.69
    },
    "WAVERLY": {
      "CLARINDA": 2.34,
      "FORT MADISON": 349.34,
      "SIOUX CITY": 246.6,
      "TOLEDO": 41.3,
      "BANCROFT": 78.73
    },
    "PELLA": {
      "CLARINDA": 1181.6,
      "FORT MADISON": 1458.53,
      "SIOUX CITY": 1646.36,
      "TOLEDO": 1924.55,
      "BANCROFT": 38.93
    },
    "DES MOINES": {
      "CLARINDA": 1030.8,
      "FORT MADISON": 43.48,
      "SIOUX CITY": 932.4299999999999,
      "TOLEDO": 55.39,
      "BANCROFT": 103.84
    }
  },
  "M": 11835
}