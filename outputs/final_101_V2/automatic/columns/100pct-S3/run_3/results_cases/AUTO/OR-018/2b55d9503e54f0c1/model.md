##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center (supplier) $i$ to customer group $j$.

Where:
- $i \in I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$
- $j \in J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

##### Parameters

Customer demands (from "customer_demand.csv"):
- $d_{\text{C1}} = 52$
- $d_{\text{C2}} = 80$
- $d_{\text{C3}} = 392$
- $d_{\text{C4}} = 103$
- $d_{\text{C5}} = 32$
- $d_{\text{C6}} = 1426$
- $d_{\text{C7}} = 1024$
- $d_{\text{C8}} = 2736$
- $d_{\text{C9}} = 1129$
- $d_{\text{C10}} = 676$
- $d_{\text{C11}} = 2631$
- $d_{\text{C12}} = 31$

Supplier capacities (from "supply_capacity.csv"):
- $s_{\text{S1}} = 58$
- $s_{\text{S2}} = 32$
- $s_{\text{S3}} = 6161$
- $s_{\text{S4}} = 4$
- $s_{\text{S5}} = 47$
- $s_{\text{S6}} = 178$
- $s_{\text{S7}} = 142$
- $s_{\text{S8}} = 164$
- $s_{\text{S9}} = 1011$
- $s_{\text{S10}} = 6$
- $s_{\text{S11}} = 7081$
- $s_{\text{S12}} = 948$

Transportation costs $c_{ij}$ (from "transportation_costs.csv", in source order):

|        | C1           | C2           | C3           | C4           | C5           | C6           | C7           | C8           | C9           | C10          | C11          | C12          |
|--------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| S1     | 134.72882438 | 72.30045141  | 37.97599274  | 611.84656498 | 1650.3353902 | 32.97044487  | 34.99837373  | 73.25207370  | 165.75208984 | 82.52035828  | 1538.8301984 | 2320.1146477 |
| S2     | 23.83536919  | 1128.6332865 | 187.05459591 | 1.60773371   | 2227.7036991 | 72.44693149  | 12.60074797  | 1078.7729548 | 383.72205694 | 63.31861812  | 82.81152504  | 1702.1724764 |
| S3     | 1138.4987596 | 231.89834225 | 44.15682267  | 962.54816407 | 692.87745740 | 980.13060324 | 143.70118867 | 33.34045088  | 136.70204402 | 1182.5161115 | 664.18467925 | 36.70238978  |
| S4     | 1043.7562803 | 24.20701606  | 1120.9890439 | 1027.5437639 | 893.44575779 | 1244.2626779 | 20.41377665  | 354.91928496 | 513.56502722 | 977.63215481 | 815.54074354 | 642.55204512 |
| S5     | 4.93993066   | 1278.1815735 | 549.75707432 | 21.33547411  | 98.32689280  | 452.09034628 | 1125.4388755 | 70.84029747  | 0.00123624   | 518.17042822 | 0.00098936   | 1783.0865677 |
| S6     | 2105.5398596 | 1340.9770559 | 2077.2747284 | 2202.2515396 | 20.53369588  | 2514.0983598 | 1280.9012839 | 683.98973438 | 1197.7655434 | 102.32440249 | 118.63491224 | 788.46403064 |
| S7     | 61.36183063  | 113.23716965 | 50.23107648  | 1219.1951077 | 869.86898644 | 58.37177230  | 92.02784777  | 552.11311210 | 168.99647982 | 1363.3427136 | 1196.7422340 | 519.38583643 |
| S8     | 1169.3640989 | 1037.2170745 | 732.35777069 | 865.22557694 | 1510.0296187 | 780.85378978 | 1118.9497799 | 935.91264328 | 61.82504910  | 906.07510075 | 808.33984318 | 58.32575132  |
| S9     | 7.93652167   | 1357.5387610 | 628.30018254 | 25.24514182  | 1760.0085250 | 604.65835357 | 1523.9047912 | 1586.4093184 | 89.68006943  | 814.76431669 | 91.00733446  | 1580.7262556 |
| S10    | 1685.3409759 | 437.39384913 | 1568.8939936 | 1486.9268968 | 498.37544466 | 1493.3860214 | 442.51135717 | 70.17952879  | 526.99999225 | 1527.6606597 | 1243.2102449 | 2.67641357   |
| S11    | 937.10940783 | 903.01414158 | 264.72089986 | 21.82334018  | 1661.1054863 | 18.59349094  | 1046.6836915 | 956.69555987 | 42.91201853  | 413.30844971 | 48.42192170  | 1274.3227527 |
| S12    | 1685.7576851 | 377.32566729 | 1347.0162725 | 1737.0200305 | 23.61217798  | 83.08521829  | 320.72681719 | 11.77469761  | 530.00219271 | 1782.8442633 | 97.99901498  | 1965.9423692 |

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$
where $c_{ij}$ are the coefficients from the table above.

##### Constraints

1. **Demand satisfaction** (each customer group must receive at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   $$
   Explicitly, for each $j$:
   - $\sum_{i} x_{i,\text{C1}} \geq 52$
   - $\sum_{i} x_{i,\text{C2}} \geq 80$
   - $\sum_{i} x_{i,\text{C3}} \geq 392$
   - $\sum_{i} x_{i,\text{C4}} \geq 103$
   - $\sum_{i} x_{i,\text{C5}} \geq 32$
   - $\sum_{i} x_{i,\text{C6}} \geq 1426$
   - $\sum_{i} x_{i,\text{C7}} \geq 1024$
   - $\sum_{i} x_{i,\text{C8}} \geq 2736$
   - $\sum_{i} x_{i,\text{C9}} \geq 1129$
   - $\sum_{i} x_{i,\text{C10}} \geq 676$
   - $\sum_{i} x_{i,\text{C11}} \geq 2631$
   - $\sum_{i} x_{i,\text{C12}} \geq 31$

2. **Supply capacity** (each supplier cannot ship more than its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   $$
   Explicitly, for each $i$:
   - $\sum_{j} x_{\text{S1},j} \leq 58$
   - $\sum_{j} x_{\text{S2},j} \leq 32$
   - $\sum_{j} x_{\text{S3},j} \leq 6161$
   - $\sum_{j} x_{\text{S4},j} \leq 4$
   - $\sum_{j} x_{\text{S5},j} \leq 47$
   - $\sum_{j} x_{\text{S6},j} \leq 178$
   - $\sum_{j} x_{\text{S7},j} \leq 142$
   - $\sum_{j} x_{\text{S8},j} \leq 164$
   - $\sum_{j} x_{\text{S9},j} \leq 1011$
   - $\sum_{j} x_{\text{S10},j} \leq 6$
   - $\sum_{j} x_{\text{S11},j} \leq 7081$
   - $\sum_{j} x_{\text{S12},j} \leq 948$

3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

##### Complete Model

Minimize
$$
\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$
subject to
$$
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
$$
$$
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
$$
$$
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
$$

with all $c_{ij}$, $d_j$, and $s_i$ as specified above.