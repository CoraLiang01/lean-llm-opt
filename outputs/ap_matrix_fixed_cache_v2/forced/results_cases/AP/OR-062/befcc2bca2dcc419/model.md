##### Objective Function:

$\quad \min \left( \sum_{i \in F} \text{fixed\_cost}_i \cdot y_i + \sum_{i \in F} \sum_{j \in S} \text{transport\_cost}_{ij} \cdot x_{ij} \right)$

##### Constraints:

1. **Demand Satisfaction:**

$\sum_{i \in F} x_{ij} = \text{demand}_j \quad \forall j \in S$

2. **Supplier Activation:**

$\sum_{j \in S} x_{ij} \leq M \cdot y_i \quad \forall i \in F$

where $M$ is a sufficiently large constant (e.g., $M = \sum_{j \in S} \text{demand}_j$).

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

- Suppliers $F$: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- Stores $S$: CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT
- Fixed costs: as above
- Demands: Customer_1 (2397), Customer_2 (1889), Customer_3 (2518), Customer_4 (3218), Customer_5 (1813)
- Transportation costs: as above

##### Variable Definitions

- $y_i$: Binary variable, $1$ if supplier $i$ is open, $0$ otherwise.
- $x_{ij}$: Quantity supplied from supplier $i$ to store $j$.

##### Notes

- Each store's demand must be fully met.
- Suppliers incur a fixed cost if activated.
- Transportation costs are per unit shipped.
- No explicit supplier capacity is given; only activation and demand constraints are enforced.