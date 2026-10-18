##### Decision Variables

$x_{ij}\geq0$: quantity of goods shipped from supplier $i\in I$ to store $j\in J$ (continuous).  
$y_i\in\{0,1\}$: binary variable indicating whether supplier $i$ is operational (open).

##### Objective Function

$\displaystyle \min\;\sum_{i\in I}f_i\,y_i+\sum_{i\in I}\sum_{j\in J}c_{ij}\,x_{ij}$

##### Constraints

1. Demand satisfaction:  
   $\displaystyle \sum_{i\in I}x_{ij}\;\geq\;d_j,\quad\forall j\in J$.

2. Linking (no shipment if supplier closed):  
   $\displaystyle x_{ij}\;\leq\;M\,y_i,\quad\forall i\in I,\;\forall j\in J$  
   where $M=11835$ (sum of all demands).

3. Binary supplier activation:  
   $\displaystyle y_i\in\{0,1\},\quad\forall i\in I$.

4. Non‑negativity:  
   $\displaystyle x_{ij}\geq0,\quad\forall i\in I,\;\forall j\in J$.

$I=\{\text{MOUNT AYR},\text{WAUKEE},\text{WAVERLY},\text{PELLA},\text{DES MOINES}\}$  
$J=\{\text{Customer\_1},\text{Customer\_2},\text{Customer\_3},\text{Customer\_4},\text{Customer\_5}\}$

###### Retrieved Information

{
  "suppliers": [
    "MOUNT AYR",
    "WAUKEE",
    "WAVERLY",
    "PELLA",
    "DES MOINES"
  ],
  "customers": [
    "Customer_1",
    "Customer_2",
    "Customer_3",
    "Customer_4",
    "Customer_5"
  ],
  "demand": {
    "Customer_1": 2397,
    "Customer_2": 1889,
    "Customer_3": 2518,
    "Customer_4": 3218,
    "Customer_5": 1813
  },
  "fixed_costs": {
    "MOUNT AYR": 96.58,
    "WAUKEE": 94.06,
    "WAVERLY": 94.37,
    "PELLA": 82.88,
    "DES MOINES": 94.95999999999999
  },
  "cost": {
    "MOUNT AYR": {
      "BANCROFT": 1685.53,
      "CLARINDA": 694.6799999999999,
      "FORT MADISON": 17.48,
      "SIOUX CITY": 20.07,
      "TOLEDO": 199.02
    },
    "WAUKEE": {
      "BANCROFT": 90.69,
      "CLARINDA": 15.13,
      "FORT MADISON": 1.5,
      "SIOUX CITY": 1.43,
      "TOLEDO": 27.88
    },
    "WAVERLY": {
      "BANCROFT": 78.73,
      "CLARINDA": 2.34,
      "FORT MADISON": 349.34,
      "SIOUX CITY": 246.6,
      "TOLEDO": 41.3
    },
    "PELLA": {
      "BANCROFT": 38.93,
      "CLARINDA": 1181.6,
      "FORT MADISON": 1458.53,
      "SIOUX CITY": 1646.36,
      "TOLEDO": 1924.55
    },
    "DES MOINES": {
      "BANCROFT": 103.84,
      "CLARINDA": 1030.8,
      "FORT MADISON": 43.48,
      "SIOUX CITY": 932.4299999999999,
      "TOLEDO": 55.39
    }
  }
}