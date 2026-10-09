Let $x_{ij}$ denote the quantity of goods transported from Walmart store $i$ to customer group $j$.

**Sets and Indices:**
- $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}\}$ (Walmart stores)
- $j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$ (Customer groups)

**Parameters:**
- $c_{ij}$: Transportation cost per unit from store $i$ to customer $j$ (see table below)
- $d_j$: Demand of customer $j$ (see table below)
- $s_i$: Supply capacity of store $i$ (see table below)

**Objective:**
\[
\min \sum_{i} \sum_{j} c_{ij} x_{ij}
\]

**Subject to:**

1. **Demand satisfaction for each customer group:**
   \[
   \sum_{i} x_{ij} = d_j \qquad \forall j
   \]

2. **Supply capacity for each Walmart store:**
   \[
   \sum_{j} x_{ij} \leq s_i \qquad \forall i
   \]

3. **Nonnegativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i, j
   \]

---

**Data:**

- Customer Demands ($d_j$):

| Customer | Demand |
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

- Walmart Store Supply Capacities ($s_i$):

| Store | Supply Capacity |
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

- Transportation Costs ($c_{ij}$):

| Store/Customer |   C1   |    C2   |    C3   |    C4   |    C5   |    C6   |    C7   |    C8   |    C9   |   C10   |   C11   |   C12   |
|:--------------:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| S1             | 0.6391 | 49.7184 | 33.7586 | 1570.67 | 1370.41 | 57.3531 | 57.1830 | 54.9210 | 1143.68 | 52.4913 | 606.443 | 1192.47 |
| S2             | 605.48 | 64.5356 | 478.478 | 887.048 | 65.4611 | 71.9361 | 41.2902 | 70.3604 | 35.3589 | 1472.75 | 0.6005  | 49.8685 |
| S3             | 1139.04| 4.7851  | 1805.62 | 1302.90 | 2437.32 | 103.804 | 774.656 | 4.5160  | 879.705 | 162.706 | 1208.61 | 110.187 |
| S4             | 69.2699| 2105.49 | 869.682 | 1494.90 | 310.538 | 98.1546 | 103.369 | 1758.88 | 97.2854 | 94.6504 | 1277.25 | 21.6362 |
| S5             | 980.411| 899.311 | 1183.03 | 402.099 | 81.7886 | 1115.68 | 123.804 | 1121.15 | 0.0024  | 1009.65 | 35.3480 | 1625.43 |
| S6             | 1246.78| 2105.80 | 1014.34 | 1494.67 | 362.017 | 98.1714 | 2170.41 | 97.7319 | 97.2683 | 1987.99 | 70.9440 | 389.160 |
| S7             | 57.1086| 23.8362 | 78.1057 | 742.807 | 1926.08 | 454.379 | 458.290 | 465.931 | 28.1386 | 524.615 | 997.532 | 104.478 |
| S8             | 981.291| 120.901 | 1625.82 | 1267.82 | 2569.64 | 13.4718 | 815.153 | 253.423 | 43.7656 | 275.978 | 1228.07 | 103.483 |
| S9             | 30.5328| 1444.86 | 173.555 | 1307.39 | 965.201 | 1843.78 | 1483.64 | 85.3221 | 1353.50 | 1485.92 | 29.4238 | 26.6194 |
| S10            | 94.1109| 1422.99 | 1470.78 | 1419.34 | 38.9453 | 72.2011 | 2040.46 | 1542.70 | 1803.80 | 72.9437 | 2181.45 | 973.552 |
| S11            | 1032.91| 166.302 | 1620.48 | 64.6683 | 2000.51 | 0.0029  | 47.0384 | 52.9922 | 1115.63 | 129.793 | 1295.10 | 2330.77 |

**Variable domains:**
\[
x_{ij} \geq 0 \quad \text{and continuous} \qquad \forall i, j
\]

**Summary:**
Minimize total transportation cost by choosing nonnegative shipment quantities $x_{ij}$ from each Walmart store $i$ to each customer group $j$, such that all customer demands are exactly met and no store's supply capacity is exceeded.