##### Decision Variables:

Let $x_i$ denote the number of units of TABLET model $i$ to be fulfilled, for each model $i$ in the set of TABLET products.

##### Objective Function:

$\quad \max \sum_{i} r_i x_i$

where $r_i$ is the revenue per unit for model $i$.

##### Constraints:

1. Inventory and Demand Constraints:

$\quad 0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\} \quad \forall i$

or, equivalently, for each model $i$:

$\quad 0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

2. Integrality Constraints:

$\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i$

##### Retrieved Information

{
  "models": [
    "TABLET_10084.74",
    "TABLET_12211.86",
    "TABLET_14669.5",
    "TABLET_14745.76",
    "TABLET_14754.24",
    "TABLET_16448.3",
    "TABLET_16448.31",
    "TABLET_20143.22",
    "TABLET_2042.38",
    "TABLET_24915.25",
    "TABLET_24915.26",
    "TABLET_26448.3",
    "TABLET_27042.38",
    "TABLET_30000.0",
    "TABLET_33397.46",
    "TABLET_33398.3",
    "TABLET_48567.8",
    "TABLET_48644.07",
    "TABLET_50262.72",
    "TABLET_53736.44",
    "TABLET_6957.62",
    "TABLET_6957.63",
    "TABLET_7550.84",
    "TABLET_7550.85",
    "TABLET_9584.74",
    "TABLET_9661.02",
    "TABLET_9669.5"
  ],
  "revenue": {
    "TABLET_10084.74": 10084.74,
    "TABLET_12211.86": 12211.86,
    "TABLET_14669.5": 14669.5,
    "TABLET_14745.76": 14745.76,
    "TABLET_14754.24": 14754.24,
    "TABLET_16448.3": 16448.3,
    "TABLET_16448.31": 16448.31,
    "TABLET_20143.22": 20143.22,
    "TABLET_2042.38": 2042.38,
    "TABLET_24915.25": 24915.25,
    "TABLET_24915.26": 24915.26,
    "TABLET_26448.3": 26448.3,
    "TABLET_27042.38": 27042.38,
    "TABLET_30000.0": 30000.0,
    "TABLET_33397.46": 33397.46,
    "TABLET_33398.3": 33398.3,
    "TABLET_48567.8": 48567.8,
    "TABLET_48644.07": 48644.07,
    "TABLET_50262.72": 50262.72,
    "TABLET_53736.44": 53736.44,
    "TABLET_6957.62": 6957.62,
    "TABLET_6957.63": 6957.63,
    "TABLET_7550.84": 7550.84,
    "TABLET_7550.85": 7550.85,
    "TABLET_9584.74": 9584.74,
    "TABLET_9661.02": 9661.02,
    "TABLET_9669.5": 9669.5
  },
  "initial_inventory": {
    "TABLET_10084.74": 10,
    "TABLET_12211.86": 300,
    "TABLET_14669.5": 30,
    "TABLET_14745.76": 30,
    "TABLET_14754.24": 100,
    "TABLET_16448.3": 110,
    "TABLET_16448.31": 20,
    "TABLET_20143.22": 80,
    "TABLET_2042.38": 10,
    "TABLET_24915.25": 10,
    "TABLET_24915.26": 160,
    "TABLET_26448.3": 70,
    "TABLET_27042.38": 10,
    "TABLET_30000.0": 10,
    "TABLET_33397.46": 30,
    "TABLET_33398.3": 10,
    "TABLET_48567.8": 30,
    "TABLET_48644.07": 10,
    "TABLET_50262.72": 30,
    "TABLET_53736.44": 30,
    "TABLET_6957.62": 40,
    "TABLET_6957.63": 80,
    "TABLET_7550.84": 300,
    "TABLET_7550.85": 40,
    "TABLET_9584.74": 40,
    "TABLET_9661.02": 190,
    "TABLET_9669.5": 20
  },
  "demand": {
    "TABLET_10084.74": 2,
    "TABLET_12211.86": 43,
    "TABLET_14669.5": 6,
    "TABLET_14745.76": 6,
    "TABLET_14754.24": 20,
    "TABLET_16448.3": 22,
    "TABLET_16448.31": 3,
    "TABLET_20143.22": 16,
    "TABLET_2042.38": 2,
    "TABLET_24915.25": 2,
    "TABLET_24915.26": 32,
    "TABLET_26448.3": 14,
    "TABLET_27042.38": 2,
    "TABLET_30000.0": 2,
    "TABLET_33397.46": 6,
    "TABLET_33398.3": 2,
    "TABLET_48567.8": 6,
    "TABLET_48644.07": 2,
    "TABLET_50262.72": 6,
    "TABLET_53736.44": 6,
    "TABLET_6957.62": 6,
    "TABLET_6957.63": 12,
    "TABLET_7550.84": 60,
    "TABLET_7550.85": 8,
    "TABLET_9584.74": 8,
    "TABLET_9661.02": 38,
    "TABLET_9669.5": 4
  }
}