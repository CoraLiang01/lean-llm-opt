Let $x_{ij}$ denote the number of units transported from supplier $i$ to customer group $j$.

**Sets:**
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

|        |  C1         |  C2         |  C3         |  C4         |  C5         |  C6         |  C7         |  C8         |  C9         |  C10        |
|--------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| S1     | 590.3648137 | 23.66917261 | 88.89005870 | 497.5222881 | 466.0903432 | 29.02209683 | 23.67524483 | 23.67776029 | 0.3118394915| 58.89547392 |
| S2     |2042.0715002 |2133.9784843 | 705.1591203 | 101.5945452 |2052.9376574 |1738.7549514 |101.61094966 |101.61062174 |122.45214269 | 67.29170751 |
| S3     | 22.29722216 | 497.9271939 |1653.0828862 | 23.68545123 |1386.0807887 | 26.13715281 |497.62204829 |498.09358471 |865.38162963 |1008.6717395 |
| S4     | 960.7814534 | 49.12830005 |1324.2386971 |1032.2095478 | 0.078047254 | 53.30826873 | 49.13641672 |1031.8214425 |466.00495308 |1351.8189071 |
| S5     |1471.2721666 | 85.69560728 | 38.89266824 |1542.0500358 |112.20514372 | 82.37020164 |1542.3399197 | 85.69238746 |1924.93607696|1094.6695961 |
| S6     | 191.9058726 |158.50401032 | 91.02045350 | 184.4474720 | 968.1467987 |284.10760621 | 8.791061588 |158.70523836 | 27.94387435 | 929.8071683 |
| S7     | 81.23891457 | 0.374464222 |2079.4668654 | 0.306567176 |1031.7772962 | 7.203964492 | 0.076230722 | 0.032473880 | 23.68582797 | 849.9799407 |
| S8     | 56.09931097 |935.61431087 | 73.08824617 | 52.00392409 | 4.025792389 |1002.2327658 |935.77660296 |935.70072523 |612.86987193 |1348.8366146 |
| S9     | 4.502283327 | 0.389958534 |1782.4662178 | 0.006345907 |1031.9910115 |129.50665620 | 0.211831957 | 0.645730107 |497.62723911 | 40.46575555 |
| S10    |333.68692704 |277.47193861 | 86.02096892 |277.30836609 |1004.4649085 | 19.95033682 | 13.20207369 |238.14321522 |411.05803324 | 941.7526366 |

**Decision Variables:**
- $x_{ij} \geq 0$, continuous (amount shipped from supplier $i$ to customer $j$)

---

### Mathematical Model

**Objective:**
\[
\min \sum_{i \in \{\text{S1},\ldots,\text{S10}\}} \sum_{j \in \{\text{C1},\ldots,\text{C10}\}} c_{ij} \, x_{ij}
\]

**Subject to:**

**1. Supply capacity constraints (for each supplier):**
\[
\sum_{j \in \{\text{C1},\ldots,\text{C10}\}} x_{ij} \leq \text{supply\_capacity}_i \qquad \forall i \in \{\text{S1},\ldots,\text{S10}\}
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

**2. Demand satisfaction constraints (for each customer group):**
\[
\sum_{i \in \{\text{S1},\ldots,\text{S10}\}} x_{ij} = \text{demand}_j \qquad \forall j \in \{\text{C1},\ldots,\text{C10}\}
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
x_{ij} \geq 0 \qquad \forall i, j
\]

---

**All parameters and indices are as retrieved and preserved in source order.**