##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Parameters

- Suppliers $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- Stores $J = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$
- Store demand:
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$
- Supplier fixed costs:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$
- Transportation costs $c_{ij}$ (supplier $i$, store $j$):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

However, the store names in demand.csv are Customer_1, ..., Customer_5. To match, we assume the following mapping (in order of appearance):

- Customer_1 $\rightarrow$ CLARINDA
- Customer_2 $\rightarrow$ FORT MADISON
- Customer_3 $\rightarrow$ SIOUX CITY
- Customer_4 $\rightarrow$ TOLEDO
- Customer_5 $\rightarrow$ BANCROFT

So, the cost matrix $c_{ij}$ is:

|               | Customer_1 (CLARINDA) | Customer_2 (FORT MADISON) | Customer_3 (SIOUX CITY) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
|---------------|-----------------------|---------------------------|-------------------------|---------------------|-----------------------|
| MOUNT AYR     | 694.68                | 17.48                     | 20.07                   | 199.02              | 1685.53               |
| WAUKEE        | 15.13                 | 1.5                       | 1.43                    | 27.88               | 90.69                 |
| WAVERLY       | 2.34                  | 349.34                    | 246.6                   | 41.3                | 78.73                 |
| PELLA         | 1181.6                | 1458.53                   | 1646.36                 | 1924.55             | 38.93                 |
| DES MOINES    | 1030.8                | 43.48                     | 932.43                  | 55.39               | 103.84                |

Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$.

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
3. **Variable domains:**
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Full Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

Where:

- $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- $J = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$
- $d_j$ as above
- $f_i$ as above
- $c_{ij}$ as above
- $M = 11835$

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
  "fixed_cost": {
    "MOUNT AYR": 96.58,
    "WAUKEE": 94.06,
    "WAVERLY": 94.37,
    "PELLA": 82.88,
    "DES MOINES": 94.96
  },
  "cost": {
    "MOUNT AYR": {
      "Customer_1": 694.68,
      "Customer_2": 17.48,
      "Customer_3": 20.07,
      "Customer_4": 199.02,
      "Customer_5": 1685.53
    },
    "WAUKEE": {
      "Customer_1": 15.13,
      "Customer_2": 1.5,
      "Customer_3": 1.43,
      "Customer_4": 27.88,
      "Customer_5": 90.69
    },
    "WAVERLY": {
      "Customer_1": 2.34,
      "Customer_2": 349.34,
      "Customer_3": 246.6,
      "Customer_4": 41.3,
      "Customer_5": 78.73
    },
    "PELLA": {
      "Customer_1": 1181.6,
      "Customer_2": 1458.53,
      "Customer_3": 1646.36,
      "Customer_4": 1924.55,
      "Customer_5": 38.93
    },
    "DES MOINES": {
      "Customer_1": 1030.8,
      "Customer_2": 43.48,
      "Customer_3": 932.43,
      "Customer_4": 55.39,
      "Customer_5": 103.84
    }
  },
  "M": 11835
}