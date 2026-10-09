Let $x_i \geq 0$ be the production quantity (continuous, nonnegative) of product $i$ ($i \in \{P1, P2, \ldots, P111\}$).

**Parameters:**

- $p_i$: Unit profit of product $i$ (from unit_product_profits.csv)
- $a_{ji}$: Processing time required by product $i$ on device $j$ (from device_time.csv, $j \in \{A, B, C, D, E, F, G, H, I, J\}$)
- $c_j$: Monthly capacity of device $j$ (from monthly_device_capacity.csv)

---

**Objective:**

$$
\max \sum_{i = 1}^{111} p_i x_i
$$

where $p_i$ is as follows (partial list for illustration, full list from data):

- $p_{P1} = 28.55$
- $p_{P2} = 12.78$
- $p_{P3} = 45.21$
- $\ldots$
- $p_{P111} = 9.99$

---

**Constraints:**

For each device $j \in \{A, B, C, D, E, F, G, H, I, J\}$:

$$
\sum_{i = 1}^{111} a_{ji} x_i \leq c_j
$$

where $a_{ji}$ and $c_j$ are as follows (all values from the retrieved data):

- For device $A$:
  - $a_{A,P1} = 8.1$, $a_{A,P2} = 2.5$, ..., $a_{A,P111} = 8.5$
  - $c_A = 3500$
- For device $B$:
  - $a_{B,P1} = 10.5$, $a_{B,P2} = 5.2$, ..., $a_{B,P111} = 8.7$
  - $c_B = 4200$
- For device $C$:
  - $a_{C,P1} = 2.1$, $a_{C,P2} = 13.4$, ..., $a_{C,P111} = 2.9$
  - $c_C = 4500$
- For device $D$:
  - $a_{D,P1} = 5.8$, $a_{D,P2} = 1.2$, ..., $a_{D,P111} = 8.3$
  - $c_D = 2800$
- For device $E$:
  - $a_{E,P1} = 9.3$, $a_{E,P2} = 4.1$, ..., $a_{E,P111} = 6.7$
  - $c_E = 3300$
- For device $F$:
  - $a_{F,P1} = 3.8$, $a_{F,P2} = 14.2$, ..., $a_{F,P111} = 8.1$
  - $c_F = 3800$
- For device $G$:
  - $a_{G,P1} = 7.2$, $a_{G,P2} = 2.8$, ..., $a_{G,P111} = 5.9$
  - $c_G = 4100$
- For device $H$:
  - $a_{H,P1} = 11.7$, $a_{H,P2} = 6.3$, ..., $a_{H,P111} = 1.4$
  - $c_H = 3900$
- For device $I$:
  - $a_{I,P1} = 1.1$, $a_{I,P2} = 11.3$, ..., $a_{I,P111} = 7.3$
  - $c_I = 4800$
- For device $J$:
  - $a_{J,P1} = 4.6$, $a_{J,P2} = 0.2$, ..., $a_{J,P111} = 12.4$
  - $c_J = 3100$

---

**Variable domains:**

$$
x_i \geq 0 \quad \forall i \in \{P1, P2, \ldots, P111\}
$$

---

**Full Model:**

$$
\begin{align*}
\max \quad & \sum_{i=1}^{111} p_i x_i \\
\text{s.t.} \quad & \sum_{i=1}^{111} a_{A,i} x_i \leq 3500 \\
                  & \sum_{i=1}^{111} a_{B,i} x_i \leq 4200 \\
                  & \sum_{i=1}^{111} a_{C,i} x_i \leq 4500 \\
                  & \sum_{i=1}^{111} a_{D,i} x_i \leq 2800 \\
                  & \sum_{i=1}^{111} a_{E,i} x_i \leq 3300 \\
                  & \sum_{i=1}^{111} a_{F,i} x_i \leq 3800 \\
                  & \sum_{i=1}^{111} a_{G,i} x_i \leq 4100 \\
                  & \sum_{i=1}^{111} a_{H,i} x_i \leq 3900 \\
                  & \sum_{i=1}^{111} a_{I,i} x_i \leq 4800 \\
                  & \sum_{i=1}^{111} a_{J,i} x_i \leq 3100 \\
                  & x_i \geq 0 \quad \forall i \in \{P1, \ldots, P111\}
\end{align*}
$$

where all $p_i$ and $a_{ji}$ are as given in the retrieved data above.