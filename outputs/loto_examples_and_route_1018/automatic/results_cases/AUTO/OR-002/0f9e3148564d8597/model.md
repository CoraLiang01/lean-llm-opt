Let $x_{ij}$ denote the quantity of goods transported from Walmart store $i$ to customer group $j$.

**Sets and Indices:**
- $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}\}$ (Walmart stores)
- $j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$ (Customer groups)

**Parameters:**
- $d_j$: Demand of customer $j$ (from customer_demand.csv)
  - C1: 11
  - C2: 1148
  - C3: 54
  - C4: 833
  - C5: 154
  - C6: 551
  - C7: 7081
  - C8: 76
  - C9: 66
  - C10: 174
  - C11: 15
  - C12: 680
- $s_i$: Supply capacity of store $i$ (from supply_capacity.csv)
  - S1: 4
  - S2: 575
  - S3: 1504
  - S4: 178
  - S5: 228
  - S6: 50
  - S7: 3
  - S8: 6148
  - S9: 6
  - S10: 10673
  - S11: 174
- $c_{ij}$: Transportation cost per unit from store $i$ to customer $j$ (from transportation_costs.csv, see table below)

|      | C1         | C2         | C3         | C4         | C5         | C6         | C7         | C8         | C9         | C10        | C11        | C12        |
|------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| S1   | 0.63914448 | 49.7184280 | 33.7585774 | 1570.67311 | 1370.40955 | 57.3530777 | 57.1829949 | 54.9209612 | 1143.68091 | 52.4912701 | 606.443440 | 1192.46865 |
| S2   | 605.478637 | 64.5356257 | 478.477903 | 887.048074 | 65.4611125 | 71.9360522 | 41.2901539 | 70.3603821 | 35.3589300 | 1472.74819 | 0.60045915 | 49.8685402 |
| S3   | 1139.04401 | 4.78505633 | 1805.62142 | 1302.89581 | 2437.32123 | 103.803686 | 774.655824 | 4.51598828 | 879.704854 | 162.705567 | 1208.61348 | 110.186885 |
| S4   | 69.2698989 | 2105.48539 | 869.682023 | 1494.89857 | 310.537662 | 98.1545572 | 103.369185 | 1758.87838 | 97.2854071 | 94.6504089 | 1277.25145 | 21.6361903 |
| S5   | 980.411426 | 899.310883 | 1183.03255 | 402.098616 | 81.7886412 | 1115.68195 | 123.804279 | 1121.14687 | 0.00244513 | 1009.64517 | 35.3480183 | 1625.43463 |
| S6   | 1246.78250 | 2105.79667 | 1014.33932 | 1494.66814 | 362.017391 | 98.1714242 | 2170.40587 | 97.7318771 | 97.2683402 | 1987.99078 | 70.9439691 | 389.159804 |
| S7   | 57.1086015 | 23.8361689 | 78.1057298 | 742.806811 | 1926.07968 | 454.378996 | 458.290144 | 465.930766 | 28.1386071 | 524.615426 | 997.531784 | 104.477945 |
| S8   | 981.290861 | 120.901300 | 1625.82069 | 1267.82293 | 2569.64461 | 13.4718118 | 815.152543 | 253.423496 | 43.7656295 | 275.978413 | 1228.06989 | 103.483230 |
| S9   | 30.5327795 | 1444.85950 | 173.554732 | 1307.39131 | 965.201230 | 1843.77695 | 1483.64088 | 85.3220995 | 1353.50093 | 1485.91538 | 29.4237908 | 26.6194196 |
| S10  | 94.1109396 | 1422.99713 | 1470.77691 | 1419.33823 | 38.9452778 | 72.2011295 | 2040.46059 | 1542.70256 | 1803.80017 | 72.9436583 | 2181.45421 | 973.551555 |
| S11  | 1032.90738 | 166.301848 | 1620.47671 | 64.6683413 | 2000.50917 | 0.00289580 | 47.0383714 | 52.9922113 | 1115.63362 | 129.793382 | 1295.09784 | 2330.76821 |

**Decision Variables:**
- $x_{ij} \geq 0$, continuous (quantity shipped from store $i$ to customer $j$)

---

### Mathematical Model

**Objective:**
\[
\min \sum_{i \in \{\text{S1},\ldots,\text{S11}\}} \sum_{j \in \{\text{C1},\ldots,\text{C12}\}} c_{ij} \cdot x_{ij}
\]

**Subject to:**

1. **Demand Satisfaction (for each customer $j$):**
   \[
   \sum_{i \in \{\text{S1},\ldots,\text{S11}\}} x_{ij} = d_j \qquad \forall j \in \{\text{C1},\ldots,\text{C12}\}
   \]
   - C1: $\sum_{i} x_{i,\text{C1}} = 11$
   - C2: $\sum_{i} x_{i,\text{C2}} = 1148$
   - C3: $\sum_{i} x_{i,\text{C3}} = 54$
   - C4: $\sum_{i} x_{i,\text{C4}} = 833$
   - C5: $\sum_{i} x_{i,\text{C5}} = 154$
   - C6: $\sum_{i} x_{i,\text{C6}} = 551$
   - C7: $\sum_{i} x_{i,\text{C7}} = 7081$
   - C8: $\sum_{i} x_{i,\text{C8}} = 76$
   - C9: $\sum_{i} x_{i,\text{C9}} = 66$
   - C10: $\sum_{i} x_{i,\text{C10}} = 174$
   - C11: $\sum_{i} x_{i,\text{C11}} = 15$
   - C12: $\sum_{i} x_{i,\text{C12}} = 680$

2. **Supply Capacity (for each store $i$):**
   \[
   \sum_{j \in \{\text{C1},\ldots,\text{C12}\}} x_{ij} \leq s_i \qquad \forall i \in \{\text{S1},\ldots,\text{S11}\}
   \]
   - S1: $\sum_{j} x_{\text{S1},j} \leq 4$
   - S2: $\sum_{j} x_{\text{S2},j} \leq 575$
   - S3: $\sum_{j} x_{\text{S3},j} \leq 1504$
   - S4: $\sum_{j} x_{\text{S4},j} \leq 178$
   - S5: $\sum_{j} x_{\text{S5},j} \leq 228$
   - S6: $\sum_{j} x_{\text{S6},j} \leq 50$
   - S7: $\sum_{j} x_{\text{S7},j} \leq 3$
   - S8: $\sum_{j} x_{\text{S8},j} \leq 6148$
   - S9: $\sum_{j} x_{\text{S9},j} \leq 6$
   - S10: $\sum_{j} x_{\text{S10},j} \leq 10673$
   - S11: $\sum_{j} x_{\text{S11},j} \leq 174$

3. **Nonnegativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i, j
   \]

---

**All coefficients and identifiers are as retrieved and preserved in source order.**