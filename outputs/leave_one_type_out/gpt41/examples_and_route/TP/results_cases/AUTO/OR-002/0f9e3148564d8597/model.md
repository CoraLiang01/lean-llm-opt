Let $x_{s,c}$ denote the quantity transported from Walmart store $s$ to customer group $c$.

**Parameters:**

- $S$ = set of Walmart stores:  
  S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11

- $C$ = set of customer groups:  
  C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12

- $demand_c$ = demand of customer group $c$ (from customer_demand.csv):

  | customer | demand |
  |----------|--------|
  | C1       | 11     |
  | C2       | 1148   |
  | C3       | 54     |
  | C4       | 833    |
  | C5       | 154    |
  | C6       | 551    |
  | C7       | 7081   |
  | C8       | 76     |
  | C9       | 66     |
  | C10      | 174    |
  | C11      | 15     |
  | C12      | 680    |

- $supply\_capacity_s$ = supply capacity of store $s$ (from supply_capacity.csv):

  | store | supply_capacity |
  |-------|----------------|
  | S1    | 4              |
  | S2    | 575            |
  | S3    | 1504           |
  | S4    | 178            |
  | S5    | 228            |
  | S6    | 50             |
  | S7    | 3              |
  | S8    | 6148           |
  | S9    | 6              |
  | S10   | 10673          |
  | S11   | 174            |

- $cost_{s,c}$ = transportation cost per unit from store $s$ to customer $c$ (from transportation_costs.csv):

  |        |   C1   |    C2   |    C3   |    C4   |    C5   |    C6   |    C7   |    C8   |    C9   |   C10   |   C11   |   C12   |
  |--------|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
  | S1     | 0.6391 | 49.7184 | 33.7586 | 1570.67 | 1370.41 | 57.3531 | 57.1830 | 54.9210 | 1143.68 | 52.4913 | 606.443 | 1192.47 |
  | S2     | 605.48 | 64.5356 | 478.478 | 887.048 | 65.4611 | 71.9361 | 41.2902 | 70.3604 | 35.3589 | 1472.75 | 0.6005  | 49.8685 |
  | S3     | 1139.0 | 4.78506 | 1805.62 | 1302.90 | 2437.32 | 103.804 | 774.656 | 4.51599 | 879.705 | 162.706 | 1208.61 | 110.187 |
  | S4     | 69.2699| 2105.49 | 869.682 | 1494.90 | 310.538 | 98.1546 | 103.369 | 1758.88 | 97.2854 | 94.6504 | 1277.25 | 21.6362 |
  | S5     | 980.411| 899.311 | 1183.03 | 402.099 | 81.7886 | 1115.68 | 123.804 | 1121.15 | 0.00245 | 1009.65 | 35.3480 | 1625.43 |
  | S6     | 1246.78| 2105.80 | 1014.34 | 1494.67 | 362.017 | 98.1714 | 2170.41 | 97.7319 | 97.2683 | 1987.99 | 70.9440 | 389.160 |
  | S7     | 57.1086| 23.8362 | 78.1057 | 742.807 | 1926.08 | 454.379 | 458.290 | 465.931 | 28.1386 | 524.615 | 997.532 | 104.478 |
  | S8     | 981.291| 120.901 | 1625.82 | 1267.82 | 2569.64 | 13.4718 | 815.153 | 253.423 | 43.7656 | 275.978 | 1228.07 | 103.483 |
  | S9     | 30.5328| 1444.86 | 173.555 | 1307.39 | 965.201 | 1843.78 | 1483.64 | 85.3221 | 1353.50 | 1485.92 | 29.4238 | 26.6194 |
  | S10    | 94.1109| 1422.99 | 1470.78 | 1419.34 | 38.9453 | 72.2011 | 2040.46 | 1542.70 | 1803.80 | 72.9437 | 2181.45 | 973.552 |
  | S11    | 1032.91| 166.302 | 1620.48 | 64.6683 | 2000.51 | 0.00290 | 47.0384 | 52.9922 | 1115.63 | 129.793 | 1295.10 | 2330.77 |

**Decision Variables:**

- $x_{s,c} \geq 0$, continuous (quantity shipped from store $s$ to customer $c$)

---

### Mathematical Model

**Objective:**

$$
\min \sum_{s \in S} \sum_{c \in C} cost_{s,c} \cdot x_{s,c}
$$

**Subject to:**

1. **Demand Satisfaction (for each customer group $c$):**

   $$
   \sum_{s \in S} x_{s,c} = demand_c, \quad \forall c \in C
   $$

2. **Supply Capacity (for each store $s$):**

   $$
   \sum_{c \in C} x_{s,c} \leq supply\_capacity_s, \quad \forall s \in S
   $$

3. **Nonnegativity:**

   $$
   x_{s,c} \geq 0, \quad \forall s \in S, \forall c \in C
   $$

---

**All identifiers and coefficients are as retrieved above.**