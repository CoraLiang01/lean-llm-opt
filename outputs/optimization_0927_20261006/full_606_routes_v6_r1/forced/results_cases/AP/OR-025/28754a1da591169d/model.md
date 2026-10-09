##### Objective Function:

$\quad \max \sum_{i \in \mathcal{T}} r_i x_i$

where:
- $\mathcal{T}$ is the set of all TABLET models,
- $r_i$ is the revenue per unit for TABLET model $i$,
- $x_i$ is the number of units of TABLET model $i$ fulfilled.

##### Constraints:

1. **Inventory and Demand Constraints:**

$\quad 0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\} \quad \forall i \in \mathcal{T}$

2. **Variable Domain:**

$\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \mathcal{T}$

##### Retrieved Information

{
  "TABLET_10084.74":   {"Revenue": 10084.74,  "Initial Inventory": 10,   "Demand": 2},
  "TABLET_12211.86":   {"Revenue": 12211.86,  "Initial Inventory": 300,  "Demand": 43},
  "TABLET_14669.5":    {"Revenue": 14669.5,   "Initial Inventory": 30,   "Demand": 6},
  "TABLET_14745.76":   {"Revenue": 14745.76,  "Initial Inventory": 30,   "Demand": 6},
  "TABLET_14754.24":   {"Revenue": 14754.24,  "Initial Inventory": 100,  "Demand": 20},
  "TABLET_16448.3":    {"Revenue": 16448.3,   "Initial Inventory": 110,  "Demand": 22},
  "TABLET_16448.31":   {"Revenue": 16448.31,  "Initial Inventory": 20,   "Demand": 3},
  "TABLET_20143.22":   {"Revenue": 20143.22,  "Initial Inventory": 80,   "Demand": 16},
  "TABLET_2042.38":    {"Revenue": 2042.38,   "Initial Inventory": 10,   "Demand": 2},
  "TABLET_24915.25":   {"Revenue": 24915.25,  "Initial Inventory": 10,   "Demand": 2},
  "TABLET_24915.26":   {"Revenue": 24915.26,  "Initial Inventory": 160,  "Demand": 32},
  "TABLET_26448.3":    {"Revenue": 26448.3,   "Initial Inventory": 70,   "Demand": 14},
  "TABLET_27042.38":   {"Revenue": 27042.38,  "Initial Inventory": 10,   "Demand": 2},
  "TABLET_30000.0":    {"Revenue": 30000.0,   "Initial Inventory": 10,   "Demand": 2},
  "TABLET_33397.46":   {"Revenue": 33397.46,  "Initial Inventory": 30,   "Demand": 6},
  "TABLET_33398.3":    {"Revenue": 33398.3,   "Initial Inventory": 10,   "Demand": 2},
  "TABLET_48567.8":    {"Revenue": 48567.8,   "Initial Inventory": 30,   "Demand": 6},
  "TABLET_48644.07":   {"Revenue": 48644.07,  "Initial Inventory": 10,   "Demand": 2},
  "TABLET_50262.72":   {"Revenue": 50262.72,  "Initial Inventory": 30,   "Demand": 6},
  "TABLET_53736.44":   {"Revenue": 53736.44,  "Initial Inventory": 30,   "Demand": 6},
  "TABLET_6957.62":    {"Revenue": 6957.62,   "Initial Inventory": 40,   "Demand": 6},
  "TABLET_6957.63":    {"Revenue": 6957.63,   "Initial Inventory": 80,   "Demand": 12},
  "TABLET_7550.84":    {"Revenue": 7550.84,   "Initial Inventory": 300,  "Demand": 60},
  "TABLET_7550.85":    {"Revenue": 7550.85,   "Initial Inventory": 40,   "Demand": 8},
  "TABLET_9584.74":    {"Revenue": 9584.74,   "Initial Inventory": 40,   "Demand": 8},
  "TABLET_9661.02":    {"Revenue": 9661.02,   "Initial Inventory": 190,  "Demand": 38},
  "TABLET_9669.5":     {"Revenue": 9669.5,    "Initial Inventory": 20,   "Demand": 4}
}

- $\mathcal{T} = \{$TABLET_10084.74, TABLET_12211.86, TABLET_14669.5, TABLET_14745.76, TABLET_14754.24, TABLET_16448.3, TABLET_16448.31, TABLET_20143.22, TABLET_2042.38, TABLET_24915.25, TABLET_24915.26, TABLET_26448.3, TABLET_27042.38, TABLET_30000.0, TABLET_33397.46, TABLET_33398.3, TABLET_48567.8, TABLET_48644.07, TABLET_50262.72, TABLET_53736.44, TABLET_6957.62, TABLET_6957.63, TABLET_7550.84, TABLET_7550.85, TABLET_9584.74, TABLET_9661.02, TABLET_9669.5$\}$

- For each $i \in \mathcal{T}$:
  - $r_i$ = Revenue per unit (see above)
  - Initial Inventory and Demand as above
  - $x_i$ = number of units fulfilled, integer, $0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

This model maximizes total revenue from fulfilling deterministic demand for all TABLET models, subject to inventory and demand limits.