##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$ (warehouses), $j \in J$ (stores).

##### Parameters

- $I = \{S1, S2, S3, S4, S5\}$ (warehouses)
- $J = \{D1, D2, D3, D4, D5\}$ (stores)
- Store demands:
  - $d_{D1} = 428$
  - $d_{D2} = 217$
  - $d_{D3} = 214$
  - $d_{D4} = 380$
  - $d_{D5} = 254$
- Warehouse supply capacities:
  - $s_{S1} = 428$
  - $s_{S2} = 217$
  - $s_{S3} = 214$
  - $s_{S4} = 380$
  - $s_{S5} = 254$
- Transportation costs $c_{ij}$:

|        | D1              | D2              | D3              | D4              | D5              |
|--------|-----------------|-----------------|-----------------|-----------------|-----------------|
| S1     | 269.39105880208 | 1.45373353909   | 99.60345345757  | 26.64078166310  | 9.53768895688   |
| S2     | 9.29184687679   | 10.87477843707  | 144.52609291615 | 11.42013307790  | 153.17568199278 |
| S3     | 9.67458430167   | 2.61916509597   | 100.82422491687 | 3.21219108879   | 133.84933961242 |
| S4     | 270.57498480010 | 32.50253586     | 4.68420980965   | 1.56822696865   | 9.58927599      |
| S5     | 226.03319106758 | 8.66916198083   | 65.47681316968  | 9.06876525846   | 202.65015316426 |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

That is,

\[
\min \Bigg[
\begin{aligned}
&269.39105880208\,x_{S1,D1} + 1.45373353909\,x_{S1,D2} + 99.60345345757\,x_{S1,D3} + 26.64078166310\,x_{S1,D4} + 9.53768895688\,x_{S1,D5} \\
+&9.29184687679\,x_{S2,D1} + 10.87477843707\,x_{S2,D2} + 144.52609291615\,x_{S2,D3} + 11.42013307790\,x_{S2,D4} + 153.17568199278\,x_{S2,D5} \\
+&9.67458430167\,x_{S3,D1} + 2.61916509597\,x_{S3,D2} + 100.82422491687\,x_{S3,D3} + 3.21219108879\,x_{S3,D4} + 133.84933961242\,x_{S3,D5} \\
+&270.57498480010\,x_{S4,D1} + 32.50253586\,x_{S4,D2} + 4.68420980965\,x_{S4,D3} + 1.56822696865\,x_{S4,D4} + 9.58927599\,x_{S4,D5} \\
+&226.03319106758\,x_{S5,D1} + 8.66916198083\,x_{S5,D2} + 65.47681316968\,x_{S5,D3} + 9.06876525846\,x_{S5,D4} + 202.65015316426\,x_{S5,D5}
\end{aligned}
\Bigg]
\]

##### Constraints

1. **Demand satisfaction (for each store $j$):**
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
   That is,
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
   That is,
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

##### Complete Model

\[
\begin{aligned}
\min\ &\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
&\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J \\
&\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I \\
&x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\end{aligned}
\]

where all coefficients and identifiers are as listed above.