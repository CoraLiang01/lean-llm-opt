##### Objective Function:

$\quad \min \sum_{i \in F} \text{fixed\_cost}_i \cdot y_i + \sum_{i \in F} \sum_{j \in S} \text{transport\_cost}_{ij} \cdot x_{ij}$

##### Constraints:

1. **Demand Satisfaction:**

$\sum_{i \in F} x_{ij} = \text{demand}_j \quad \forall j \in S$

2. **Supplier Activation:**

$x_{ij} \leq \text{demand}_j \cdot y_i \quad \forall i \in F, \forall j \in S$

3. **Variable Domains:**

$y_i \in \{0,1\} \quad \forall i \in F$

$x_{ij} \geq 0 \quad \forall i \in F, \forall j \in S$

---

##### Retrieved Information

```json
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
  "fixed_cost": {
    "MOUNT AYR": 96.58,
    "WAUKEE": 94.06,
    "WAVERLY": 94.37,
    "PELLA": 82.88,
    "DES MOINES": 94.96
  },
  "demand": {
    "Customer_1": 2397,
    "Customer_2": 1889,
    "Customer_3": 2518,
    "Customer_4": 3218,
    "Customer_5": 1813
  },
  "transport_cost": {
    "MOUNT AYR": {
      "CLARINDA": 694.68,
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
      "SIOUX CITY": 932.43,
      "TOLEDO": 55.39,
      "BANCROFT": 103.84
    }
  }
}
```

- Let $F$ be the set of suppliers: $\{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- Let $S$ be the set of stores: $\{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$
- Let $y_i$ be a binary variable indicating if supplier $i$ is open.
- Let $x_{ij}$ be the quantity supplied from supplier $i$ to store $j$.

##### Parameter Tables

- Fixed costs:
  - MOUNT AYR: 96.58
  - WAUKEE: 94.06
  - WAVERLY: 94.37
  - PELLA: 82.88
  - DES MOINES: 94.96

- Store demands:
  - Customer_1: 2397
  - Customer_2: 1889
  - Customer_3: 2518
  - Customer_4: 3218
  - Customer_5: 1813

- Transportation costs (per unit from supplier to store):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

##### Variable Definitions

- $y_i \in \{0,1\}$ for each supplier $i$
- $x_{ij} \geq 0$ for each supplier $i$ and store $j$

##### Model Summary

Minimize total fixed and transportation costs by choosing which suppliers to open and how much each supplies to each store, such that all store demands are met and only open suppliers can supply.