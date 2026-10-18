Let $x_{ij}$ denote the number of units transported from supplier $i$ to customer $j$.

**Sets and Indices:**
- Suppliers $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- Customers $j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

**Parameters:**
- Supply capacities:
  - S1: $288$
  - S2: $288$
  - S3: $264$
  - S4: $264$
  - S5: $216$
  - S6: $216$
  - S7: $168$
  - S8: $216$
  - S9: $240$
  - S10: $168$
- Customer demands:
  - C1: $216$
  - C2: $168$
  - C3: $264$
  - C4: $216$
  - C5: $216$
  - C6: $192$
  - C7: $144$
  - C8: $168$
  - C9: $168$
  - C10: $168$
- Transportation costs $c_{ij}$ (per unit from supplier $i$ to customer $j$):

|      | C1         | C2         | C3         | C4         | C5         | C6         | C7         | C8         | C9         | C10        |
|------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| S1   | 590.3648137| 23.6691726 | 88.8900587 | 497.5222881| 466.0903432| 29.0220968 | 23.6752448 | 23.6777603 | 0.3118395  | 58.8954739 |
| S2   |2042.0715002|2133.9784843| 705.1591203| 101.5945452|2052.9376574|1738.7549514|101.6109497 |101.6106217 |122.4521427 | 67.2917075 |
| S3   | 22.2972222 | 497.9271939|1653.0828862| 23.6854512 |1386.0807887| 26.1371528 |497.6220483 |498.0935847 |865.3816296 |1008.6717395|
| S4   | 960.7814534| 49.1283001 |1324.2386971|1032.2095478| 0.0780473  | 53.3082687 | 49.1364167 |1031.8214425|466.0049531 |1351.8189071|
| S5   |1471.2721666| 85.6956073 | 38.8926682 |1542.0500358|112.2051437 | 82.3702016 |1542.3399197| 85.6923875 |1924.9360770|1094.6695961|
| S6   |191.9058726 |158.5040103 | 91.0204535 |184.4474720 |968.1467987 |284.1076062 | 8.7910616  |158.7052384 | 27.9438743 |929.8071683 |
| S7   | 81.2389146 | 0.3744642  |2079.4668654| 0.3065672  |1031.7772962| 7.2039645  | 0.0762307  | 0.0324739  | 23.6858280 |849.9799407 |
| S8   | 56.0993110 |935.6143109 | 73.0882462 | 52.0039241 | 4.0257924  |1002.2327658|935.7766030 |935.7007252 |612.8698719 |1348.8366146|
| S9   | 4.5022833  | 0.3899585  |1782.4662178| 0.0063459  |1031.9910115|129.5066562 | 0.2118320  | 0.6457301  |497.6272391 | 40.4657555 |
| S10  |333.6869270 |277.4719386 | 86.0209689 |277.3083661 |1004.4649085| 19.9503368 | 13.2020737 |238.1432152 |411.0580332 |941.7526366 |

**Decision Variables:**
- $x_{ij} \geq 0$, continuous (amount shipped from supplier $i$ to customer $j$)

---

### Mathematical Model

**Objective:**
\[
\min \sum_{i \in \{\text{S1},\ldots,\text{S10}\}} \sum_{j \in \{\text{C1},\ldots,\text{C10}\}} c_{ij} \cdot x_{ij}
\]

**Subject to:**

**1. Supply capacity constraints (for each supplier $i$):**
\[
\sum_{j \in \{\text{C1},\ldots,\text{C10}\}} x_{ij} \leq \text{supply\_capacity}_i
\]
- S1: $\sum_j x_{\text{S1},j} \leq 288$
- S2: $\sum_j x_{\text{S2},j} \leq 288$
- S3: $\sum_j x_{\text{S3},j} \leq 264$
- S4: $\sum_j x_{\text{S4},j} \leq 264$
- S5: $\sum_j x_{\text{S5},j} \leq 216$
- S6: $\sum_j x_{\text{S6},j} \leq 216$
- S7: $\sum_j x_{\text{S7},j} \leq 168$
- S8: $\sum_j x_{\text{S8},j} \leq 216$
- S9: $\sum_j x_{\text{S9},j} \leq 240$
- S10: $\sum_j x_{\text{S10},j} \leq 168$

**2. Demand satisfaction constraints (for each customer $j$):**
\[
\sum_{i \in \{\text{S1},\ldots,\text{S10}\}} x_{ij} = \text{demand}_j
\]
- C1: $\sum_i x_{i,\text{C1}} = 216$
- C2: $\sum_i x_{i,\text{C2}} = 168$
- C3: $\sum_i x_{i,\text{C3}} = 264$
- C4: $\sum_i x_{i,\text{C4}} = 216$
- C5: $\sum_i x_{i,\text{C5}} = 216$
- C6: $\sum_i x_{i,\text{C6}} = 192$
- C7: $\sum_i x_{i,\text{C7}} = 144$
- C8: $\sum_i x_{i,\text{C8}} = 168$
- C9: $\sum_i x_{i,\text{C9}} = 168$
- C10: $\sum_i x_{i,\text{C10}} = 168$

**3. Nonnegativity:**
\[
x_{ij} \geq 0 \quad \forall i, j
\]

---

**All data used in the model:**

- Suppliers: S1, S2, S3, S4, S5, S6, S7, S8, S9, S10
- Customers: C1, C2, C3, C4, C5, C6, C7, C8, C9, C10
- Supply capacities: as listed above
- Customer demands: as listed above
- Transportation costs: as listed above (full matrix)

**Variable domains:** $x_{ij} \geq 0$, continuous

**Objective:** Minimize total transportation cost while meeting all customer demands and not exceeding any supplier's capacity.