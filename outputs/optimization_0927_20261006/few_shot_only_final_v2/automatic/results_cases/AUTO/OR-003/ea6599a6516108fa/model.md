**Sets:**

- Suppliers $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- Customers $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

**Parameters:**

- Demand $d_j$ for each customer $j$:

  - $d_{\text{C1}} = 216$
  - $d_{\text{C2}} = 168$
  - $d_{\text{C3}} = 264$
  - $d_{\text{C4}} = 216$
  - $d_{\text{C5}} = 216$
  - $d_{\text{C6}} = 192$
  - $d_{\text{C7}} = 144$
  - $d_{\text{C8}} = 168$
  - $d_{\text{C9}} = 168$
  - $d_{\text{C10}} = 168$

- Supply capacity $s_i$ for each supplier $i$:

  - $s_{\text{S1}} = 288$
  - $s_{\text{S2}} = 288$
  - $s_{\text{S3}} = 264$
  - $s_{\text{S4}} = 264$
  - $s_{\text{S5}} = 216$
  - $s_{\text{S6}} = 216$
  - $s_{\text{S7}} = 168$
  - $s_{\text{S8}} = 216$
  - $s_{\text{S9}} = 240$
  - $s_{\text{S10}} = 168$

- Transportation cost $c_{ij}$ from supplier $i$ to customer $j$:

|        |  C1         |  C2         |  C3         |  C4         |  C5         |  C6         |  C7         |  C8         |  C9         |  C10        |
|--------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| S1     | 590.3648137 | 23.66917261 | 88.89005870 | 497.5222881 | 466.0903432 | 29.02209683 | 23.67524483 | 23.67776029 | 0.3118394915| 58.89547392 |
| S2     | 2042.071500 | 2133.978484 | 705.1591203 | 101.5945452 | 2052.937657 | 1738.754951 | 101.6109497 | 101.6106217 | 122.4521427 | 67.29170751 |
| S3     | 22.29722216 | 497.9271939 | 1653.082886 | 23.68545123 | 1386.080789 | 26.13715281 | 497.6220483 | 498.0935847 | 865.3816296 | 1008.671739 |
| S4     | 960.7814534 | 49.12830005 | 1324.238697 | 1032.209548 | 0.078047254 | 53.30826873 | 49.13641672 | 1031.821442 | 466.0049531 | 1351.818907 |
| S5     | 1471.272167 | 85.69560728 | 38.89266824 | 1542.050036 | 112.2051437 | 82.37020164 | 1542.339920 | 85.69238746 | 1924.936077 | 1094.669596 |
| S6     | 191.9058726 | 158.5040103 | 91.02045350 | 184.4474720 | 968.1467987 | 284.1076062 | 8.791061588 | 158.7052384 | 27.94387435 | 929.8071683 |
| S7     | 81.23891457 | 0.374464222 | 2079.466865 | 0.306567176 | 1031.777296 | 7.203964492 | 0.076230722 | 0.032473880 | 23.68582797 | 849.9799407 |
| S8     | 56.09931097 | 935.6143109 | 73.08824617 | 52.00392409 | 4.025792389 | 1002.232766 | 935.7766030 | 935.7007252 | 612.8698719 | 1348.836615 |
| S9     | 4.502283327 | 0.389958534 | 1782.466218 | 0.006345907 | 1031.991011 | 129.5066562 | 0.211831957 | 0.645730107 | 497.6272391 | 40.46575555 |
| S10    | 333.6869270 | 277.4719386 | 86.02096892 | 277.3083661 | 1004.464908 | 19.95033682 | 13.20207369 | 238.1432152 | 411.0580332 | 941.7526366 |

**Decision Variables:**

- $x_{ij} \geq 0$ (continuous): quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$.

**Objective:**

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

**Constraints:**

1. **Demand satisfaction:** For each customer $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
   That is,
   - $\sum_{i \in I} x_{i,\text{C1}} \geq 216$
   - $\sum_{i \in I} x_{i,\text{C2}} \geq 168$
   - $\sum_{i \in I} x_{i,\text{C3}} \geq 264$
   - $\sum_{i \in I} x_{i,\text{C4}} \geq 216$
   - $\sum_{i \in I} x_{i,\text{C5}} \geq 216$
   - $\sum_{i \in I} x_{i,\text{C6}} \geq 192$
   - $\sum_{i \in I} x_{i,\text{C7}} \geq 144$
   - $\sum_{i \in I} x_{i,\text{C8}} \geq 168$
   - $\sum_{i \in I} x_{i,\text{C9}} \geq 168$
   - $\sum_{i \in I} x_{i,\text{C10}} \geq 168$

2. **Supply capacity:** For each supplier $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
   That is,
   - $\sum_{j \in J} x_{\text{S1},j} \leq 288$
   - $\sum_{j \in J} x_{\text{S2},j} \leq 288$
   - $\sum_{j \in J} x_{\text{S3},j} \leq 264$
   - $\sum_{j \in J} x_{\text{S4},j} \leq 264$
   - $\sum_{j \in J} x_{\text{S5},j} \leq 216$
   - $\sum_{j \in J} x_{\text{S6},j} \leq 216$
   - $\sum_{j \in J} x_{\text{S7},j} \leq 168$
   - $\sum_{j \in J} x_{\text{S8},j} \leq 216$
   - $\sum_{j \in J} x_{\text{S9},j} \leq 240$
   - $\sum_{j \in J} x_{\text{S10},j} \leq 168$

3. **Non-negativity:** For all $i \in I$, $j \in J$,
   $$
   x_{ij} \geq 0
   $$

**Complete Model:**

Minimize
$$
\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

with all parameters and indices as specified above.