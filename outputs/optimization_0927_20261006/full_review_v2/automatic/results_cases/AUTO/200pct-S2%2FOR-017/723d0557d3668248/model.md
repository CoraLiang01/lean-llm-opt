##### Sets

- Suppliers $I = \{S1, S2, S3, S4, S5, S6, S7, S8, S9, S10\}$
- Customers $J = \{C1, C2, C3, C4, C5, C6, C7, C8, C9, C10\}$

##### Parameters

- Demand $d_j$ for each customer $j$:
  - $d_{C1} = 216$
  - $d_{C2} = 168$
  - $d_{C3} = 264$
  - $d_{C4} = 216$
  - $d_{C5} = 216$
  - $d_{C6} = 192$
  - $d_{C7} = 144$
  - $d_{C8} = 168$
  - $d_{C9} = 168$
  - $d_{C10} = 168$

- Supply capacity $s_i$ for each supplier $i$:
  - $s_{S1} = 288$
  - $s_{S2} = 288$
  - $s_{S3} = 264$
  - $s_{S4} = 264$
  - $s_{S5} = 216$
  - $s_{S6} = 216$
  - $s_{S7} = 168$
  - $s_{S8} = 216$
  - $s_{S9} = 240$
  - $s_{S10} = 168$

- Transportation cost $c_{ij}$ for each supplier $i$ and customer $j$:

|         | C1           | C2           | C3           | C4           | C5           | C6           | C7           | C8           | C9           | C10          |
|---------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| S1      | 590.3648137  | 23.66917261  | 88.8900587   | 497.5222881  | 466.0903432  | 29.02209683  | 23.67524483  | 23.67776029  | 0.311839491  | 58.89547392  |
| S2      | 2042.0715    | 2133.978484  | 705.1591203  | 101.5945452  | 2052.937657  | 1738.754951  | 101.6109497  | 101.6106217  | 122.4521427  | 67.29170751  |
| S3      | 22.29722216  | 497.9271939  | 1653.082886  | 23.68545123  | 1386.080789  | 26.13715281  | 497.6220483  | 498.0935847  | 865.3816296  | 1008.671739  |
| S4      | 960.7814534  | 49.12830005  | 1324.238697  | 1032.209548  | 0.078047254  | 53.30826873  | 49.13641672  | 1031.821442  | 466.0049531  | 1351.818907  |
| S5      | 1471.272167  | 85.69560728  | 38.89266824  | 1542.050036  | 112.2051437  | 82.37020164  | 1542.33992   | 85.69238746  | 1924.936077  | 1094.669596  |
| S6      | 191.9058726  | 158.5040103  | 91.0204535   | 184.447472   | 968.1467987  | 284.1076062  | 8.791061588  | 158.7052384  | 27.94387435  | 929.8071683  |
| S7      | 81.23891457  | 0.374464222  | 2079.466865  | 0.306567176  | 1031.777296  | 7.203964492  | 0.076230722  | 0.03247388   | 23.68582797  | 849.9799407  |
| S8      | 56.09931097  | 935.6143109  | 73.08824617  | 52.00392409  | 4.025792389  | 1002.232766  | 935.776603   | 935.7007252  | 612.8698719  | 1348.836615  |
| S9      | 4.502283327  | 0.389958534  | 1782.466218  | 0.006345907  | 1031.991011  | 129.5066562  | 0.211831957  | 0.645730107  | 497.6272391  | 40.46575555  |
| S10     | 333.686927   | 277.4719386  | 86.02096892  | 277.3083661  | 1004.464908  | 19.95033682  | 13.20207369  | 238.1432152  | 411.0580332  | 941.7526366  |

##### Decision Variables

- $x_{ij} \geq 0$ (continuous): quantity shipped from supplier $i$ to customer $j$.

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each customer receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   $$
   Explicitly:
   - $\sum_{i \in I} x_{i,C1} \geq 216$
   - $\sum_{i \in I} x_{i,C2} \geq 168$
   - $\sum_{i \in I} x_{i,C3} \geq 264$
   - $\sum_{i \in I} x_{i,C4} \geq 216$
   - $\sum_{i \in I} x_{i,C5} \geq 216$
   - $\sum_{i \in I} x_{i,C6} \geq 192$
   - $\sum_{i \in I} x_{i,C7} \geq 144$
   - $\sum_{i \in I} x_{i,C8} \geq 168$
   - $\sum_{i \in I} x_{i,C9} \geq 168$
   - $\sum_{i \in I} x_{i,C10} \geq 168$

2. **Supply capacity** (each supplier ships no more than its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   $$
   Explicitly:
   - $\sum_{j \in J} x_{S1,j} \leq 288$
   - $\sum_{j \in J} x_{S2,j} \leq 288$
   - $\sum_{j \in J} x_{S3,j} \leq 264$
   - $\sum_{j \in J} x_{S4,j} \leq 264$
   - $\sum_{j \in J} x_{S5,j} \leq 216$
   - $\sum_{j \in J} x_{S6,j} \leq 216$
   - $\sum_{j \in J} x_{S7,j} \leq 168$
   - $\sum_{j \in J} x_{S8,j} \leq 216$
   - $\sum_{j \in J} x_{S9,j} \leq 240$
   - $\sum_{j \in J} x_{S10,j} \leq 168$

3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

##### Complete Model

Minimize
$$
\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to
$$
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
$$
$$
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
$$
$$
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
$$

Where all $c_{ij}$, $d_j$, and $s_i$ are as listed above.