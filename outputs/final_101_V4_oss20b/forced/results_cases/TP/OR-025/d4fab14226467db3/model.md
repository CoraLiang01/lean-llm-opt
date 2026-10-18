##### Decision Variables

$x_i\geq0$: number of units of tablet model $i$ that will be fulfilled, for each $i\in I$.

##### Objective Function

$\max\sum_{i\in I}r_i\,x_i$

##### Constraints

1. Demand: $\displaystyle x_i\leq d_i,\quad\forall i\in I$  
2. Initial inventory: $\displaystyle x_i\leq s_i,\quad\forall i\in I$  
3. Non‑negativity: $x_i\geq0,\quad\forall i\in I$

$I=\{\text{TABLET\_10084.74},\text{TABLET\_12211.86},\text{TABLET\_14669.5},\text{TABLET\_14745.76},\text{TABLET\_14754.24},\text{TABLET\_16448.3},\text{TABLET\_16448.31},\text{TABLET\_20143.22},\text{TABLET\_2042.38},\text{TABLET\_24915.25},\text{TABLET\_24915.26},\text{TABLET\_26448.3},\text{TABLET\_27042.38},\text{TABLET\_30000.0},\text{TABLET\_33397.46},\text{TABLET\_33398.3},\text{TABLET\_48567.8},\text{TABLET\_48644.07},\text{TABLET\_50262.72},\text{TABLET\_53736.44},\text{TABLET\_6957.62},\text{TABLET\_6957.63},\text{TABLET\_7550.84},\text{TABLET\_7550.85},\text{TABLET\_9584.74},\text{TABLET\_9661.02},\text{TABLET\_9669.5}\}$  

$d_i$ denotes the demand for product $i$, $s_i$ denotes the initial inventory for product $i$, and $r_i$ denotes the revenue per unit for product $i$.

###### Retrieved Information

{
  "products": [
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
  },
  "inventory": {
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
  }
}