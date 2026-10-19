##### Decision Variables

$x_i\geq0$: number of units of product $i$ that the store intends to fulfill, for each $i\in I$ (continuous).

##### Objective Function

$\max\sum_{i\in I}r_i\,x_i$

##### Constraints

1. Demand: $\displaystyle \sum_{i\in I}x_i\leq d_i,\quad\forall i\in I$  
2. Initial inventory: $\displaystyle \sum_{i\in I}x_i\leq s_i,\quad\forall i\in I$  
3. Non‑negativity: $x_i\geq0,\quad\forall i\in I$  

$I=\{S700\_1138,S700\_1691,S700\_1938,S700\_2047,S700\_2466,S700\_2610,S700\_2824,S700\_2834,S700\_3167,S700\_3505,S700\_3962,S700\_4002\}$  

###### Retrieved Information

{
  "products": [
    "S700_1138",
    "S700_1691",
    "S700_1938",
    "S700_2047",
    "S700_2466",
    "S700_2610",
    "S700_2824",
    "S700_2834",
    "S700_3167",
    "S700_3505",
    "S700_3962",
    "S700_4002"
  ],
  "revenue": [
    70.67,
    100.0,
    70.15,
    100.0,
    100.0,
    65.77,
    100.0,
    100.0,
    74.4,
    81.14,
    100.0,
    61.44
  ],
  "demand": [
    1219,
    1127,
    1129,
    1176,
    1301,
    1340,
    1357,
    1158,
    1287,
    1281,
    1135,
    1392
  ],
  "initial_inventory": [
    9020,
    8370,
    8390,
    8680,
    9400,
    9900,
    9760,
    8610,
    9380,
    9170,
    8520,
    10290
  ]
}