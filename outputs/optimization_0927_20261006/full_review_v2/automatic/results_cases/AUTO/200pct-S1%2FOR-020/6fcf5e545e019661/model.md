##### Sets

- Warehouses (suppliers): $I = \{S1, S2, S3, S4, S5\}$
- Stores (customers): $J = \{D1, D2, D3, D4, D5\}$

##### Parameters

- Demand for each store $j$ ($d_j$):

  - $d_{D1} = 428$
  - $d_{D2} = 217$
  - $d_{D3} = 214$
  - $d_{D4} = 380$
  - $d_{D5} = 254$

- Supply capacity for each warehouse $i$ ($s_i$):

  - $s_{S1} = 428$
  - $s_{S2} = 217$
  - $s_{S3} = 214$
  - $s_{S4} = 380$
  - $s_{S5} = 254$

- Transportation cost per unit from warehouse $i$ to store $j$ ($c_{ij}$):

  |         | D1           | D2           | D3           | D4           | D5           |
  |---------|--------------|--------------|--------------|--------------|--------------|
  | S1      | 269.3910588  | 1.453733539  | 99.60345346  | 26.64078166  | 9.537688957  |
  | S2      | 9.291846877  | 10.87477844  | 144.5260929  | 11.42013308  | 153.17568199 |
  | S3      | 9.674584302  | 2.619165096  | 100.8242249  | 3.212191089  | 133.84933961 |
  | S4      | 270.5749848  | 32.50253586  | 4.684209810  | 1.568226969  | 9.589275990  |
  | S5      | 226.0331911  | 8.669161981  | 65.47681317  | 9.068765258  | 202.65015316 |

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

That is,

\[
\min \Bigg[
\begin{aligned}
&269.3910588\,x_{S1,D1} + 1.453733539\,x_{S1,D2} + 99.60345346\,x_{S1,D3} + 26.64078166\,x_{S1,D4} + 9.537688957\,x_{S1,D5} \\
+&9.291846877\,x_{S2,D1} + 10.87477844\,x_{S2,D2} + 144.5260929\,x_{S2,D3} + 11.42013308\,x_{S2,D4} + 153.17568199\,x_{S2,D5} \\
+&9.674584302\,x_{S3,D1} + 2.619165096\,x_{S3,D2} + 100.8242249\,x_{S3,D3} + 3.212191089\,x_{S3,D4} + 133.84933961\,x_{S3,D5} \\
+&270.5749848\,x_{S4,D1} + 32.50253586\,x_{S4,D2} + 4.684209810\,x_{S4,D3} + 1.568226969\,x_{S4,D4} + 9.589275990\,x_{S4,D5} \\
+&226.0331911\,x_{S5,D1} + 8.669161981\,x_{S5,D2} + 65.47681317\,x_{S5,D3} + 9.068765258\,x_{S5,D4} + 202.65015316\,x_{S5,D5}
\end{aligned}
\Bigg]
\]

##### Constraints

1. **Demand satisfaction (for each store $j$):**
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
   Explicitly:
   \begin{align*}
   x_{S1,D1} + x_{S2,D1} + x_{S3,D1} + x_{S4,D1} + x_{S5,D1} &\geq 428 \\
   x_{S1,D2} + x_{S2,D2} + x_{S3,D2} + x_{S4,D2} + x_{S5,D2} &\geq 217 \\
   x_{S1,D3} + x_{S2,D3} + x_{S3,D3} + x_{S4,D3} + x_{S5,D3} &\geq 214 \\
   x_{S1,D4} + x_{S2,D4} + x_{S3,D4} + x_{S4,D4} + x_{S5,D4} &\geq 380 \\
   x_{S1,D5} + x_{S2,D5} + x_{S3,D5} + x_{S4,D5} + x_{S5,D5} &\geq 254 \\
   \end{align*}

2. **Supply capacity (for each warehouse $i$):**
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
   Explicitly:
   \begin{align*}
   x_{S1,D1} + x_{S1,D2} + x_{S1,D3} + x_{S1,D4} + x_{S1,D5} &\leq 428 \\
   x_{S2,D1} + x_{S2,D2} + x_{S2,D3} + x_{S2,D4} + x_{S2,D5} &\leq 217 \\
   x_{S3,D1} + x_{S3,D2} + x_{S3,D3} + x_{S3,D4} + x_{S3,D5} &\leq 214 \\
   x_{S4,D1} + x_{S4,D2} + x_{S4,D3} + x_{S4,D4} + x_{S4,D5} &\leq 380 \\
   x_{S5,D1} + x_{S5,D2} + x_{S5,D3} + x_{S5,D4} + x_{S5,D5} &\leq 254 \\
   \end{align*}

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Variable Domain

- $x_{ij} \geq 0$ and continuous for all $i \in I$, $j \in J$.