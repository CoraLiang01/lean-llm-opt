Let:

- $x_1$ = quantity of Product I produced (continuous, $\geq 0$)
- $x_2$ = quantity of Product II produced (continuous, $\geq 0$)
- $x_3$ = quantity of Product III produced (continuous, $\geq 0$)

Let $y_{ek}$ = time (in hours) equipment $e$ is used for product $k$ (continuous, $\geq 0$), for all feasible $(e,k)$ pairs.

Define:

- Equipment: $E = \{\text{A1}, \text{A2}, \text{A3}, \text{B1}, \text{B2}, \text{B3}, \text{B4}\}$
- Products: $K = \{\text{I}, \text{II}, \text{III}\}$

#### Parameters (from 43.csv, in source order):

| Equipment | Product I | Product II | Product III | Available Equipment Operating Time | Equipment Cost at Full Load (yuan) |
|-----------|-----------|------------|-------------|-----------------------------------|------------------------------------|
| A1        | 5         | 10         |             | 6000                              | 300                                |
| A2        | 7         | 9          | 12          | 10000                             | 321                                |
| A3        | 6         | 11         | 2           | 8000                              | 203                                |
| B1        | 6         | 8          |             | 4000                              | 250                                |
| B2        | 4         |            | 11          | 7000                              | 783                                |
| B3        | 7         |            |             | 4000                              | 200                                |
| B4        | 3         | 5          | 8           | 5000                              | 300                                |

- Raw Material Cost (yuan/unit): Product I: 0.25, Product II: 0.35, Product III: 0.5
- Unit Price (yuan/unit): Product I: 1.25, Product II: 2, Product III: 2.8

#### Feasible assignments (from user description):

- Product I: A1, A2, A3 for procedure A; B1, B2, B3, B4 for procedure B
- Product II: A1, A2, A3 for procedure A; B1 for procedure B
- Product III: A2, A3 for procedure A; B2, B4 for procedure B

Let $t_{ek}$ = processing time per unit of product $k$ on equipment $e$ (from table above; blank means not allowed).

#### Decision variables:

- $x_1, x_2, x_3 \geq 0$ (continuous)
- $y_{ek} \geq 0$ for all feasible $(e,k)$

#### Objective Function

Maximize total profit = total revenue - total raw material cost - total equipment cost

\[
\max \Bigg[
(1.25 - 0.25)x_1 + (2 - 0.35)x_2 + (2.8 - 0.5)x_3
- \sum_{e \in E} \frac{C_e}{T_e} \cdot \left( \sum_{k} y_{ek} \right)
\Bigg]
\]

where $C_e$ is the equipment cost at full load for $e$, $T_e$ is the available operating time for $e$.

#### Constraints

1. **Procedure A assignment (for each product):**

- Product I: $x_1 = \frac{y_{\text{A1},1}}{5} + \frac{y_{\text{A2},1}}{7} + \frac{y_{\text{A3},1}}{6}$
- Product II: $x_2 = \frac{y_{\text{A1},2}}{10} + \frac{y_{\text{A2},2}}{9} + \frac{y_{\text{A3},2}}{11}$
- Product III: $x_3 = \frac{y_{\text{A2},3}}{12} + \frac{y_{\text{A3},3}}{2}$

2. **Procedure B assignment (for each product):**

- Product I: $x_1 = \frac{y_{\text{B1},1}}{6} + \frac{y_{\text{B2},1}}{4} + \frac{y_{\text{B3},1}}{7} + \frac{y_{\text{B4},1}}{3}$
- Product II: $x_2 = \frac{y_{\text{B1},2}}{8}$
- Product III: $x_3 = \frac{y_{\text{B2},3}}{11} + \frac{y_{\text{B4},3}}{8}$

3. **Equipment time limits:**

For each equipment $e$:

\[
\sum_{k} y_{ek} \leq T_e
\]

where $T_e$ is the available operating time for $e$.

4. **Non-negativity:**

\[
x_1 \geq 0,\quad x_2 \geq 0,\quad x_3 \geq 0
\]
\[
y_{ek} \geq 0 \quad \text{for all feasible } (e,k)
\]

#### Complete Model

\[
\begin{align*}
\max\ & [1.0\, x_1 + 1.65\, x_2 + 2.3\, x_3] - \sum_{e \in E} \frac{C_e}{T_e} \left( \sum_{k} y_{ek} \right) \\
\text{s.t.} \\
& x_1 = \frac{y_{\text{A1},1}}{5} + \frac{y_{\text{A2},1}}{7} + \frac{y_{\text{A3},1}}{6} \\
& x_2 = \frac{y_{\text{A1},2}}{10} + \frac{y_{\text{A2},2}}{9} + \frac{y_{\text{A3},2}}{11} \\
& x_3 = \frac{y_{\text{A2},3}}{12} + \frac{y_{\text{A3},3}}{2} \\
& x_1 = \frac{y_{\text{B1},1}}{6} + \frac{y_{\text{B2},1}}{4} + \frac{y_{\text{B3},1}}{7} + \frac{y_{\text{B4},1}}{3} \\
& x_2 = \frac{y_{\text{B1},2}}{8} \\
& x_3 = \frac{y_{\text{B2},3}}{11} + \frac{y_{\text{B4},3}}{8} \\
& y_{\text{A1},1} + y_{\text{A1},2} \leq 6000 \\
& y_{\text{A2},1} + y_{\text{A2},2} + y_{\text{A2},3} \leq 10000 \\
& y_{\text{A3},1} + y_{\text{A3},2} + y_{\text{A3},3} \leq 8000 \\
& y_{\text{B1},1} + y_{\text{B1},2} \leq 4000 \\
& y_{\text{B2},1} + y_{\text{B2},3} \leq 7000 \\
& y_{\text{B3},1} \leq 4000 \\
& y_{\text{B4},1} + y_{\text{B4},2} + y_{\text{B4},3} \leq 5000 \\
& x_1, x_2, x_3 \geq 0 \\
& y_{ek} \geq 0 \quad \text{for all feasible } (e,k)
\end{align*}
\]

where the only nonzero $y_{ek}$ are for the following $(e,k)$ pairs (all others are fixed to zero):

- $y_{\text{A1},1}$, $y_{\text{A1},2}$
- $y_{\text{A2},1}$, $y_{\text{A2},2}$, $y_{\text{A2},3}$
- $y_{\text{A3},1}$, $y_{\text{A3},2}$, $y_{\text{A3},3}$
- $y_{\text{B1},1}$, $y_{\text{B1},2}$
- $y_{\text{B2},1}$, $y_{\text{B2},3}$
- $y_{\text{B3},1}$
- $y_{\text{B4},1}$, $y_{\text{B4},2}$, $y_{\text{B4},3}$

Equipment costs per hour (for the penalty term):

- A1: $300/6000 = 0.05$
- A2: $321/10000 = 0.0321$
- A3: $203/8000 = 0.025375$
- B1: $250/4000 = 0.0625$
- B2: $783/7000 = 0.111857$
- B3: $200/4000 = 0.05$
- B4: $300/5000 = 0.06$

So the equipment cost term is:

\[
0.05(y_{\text{A1},1} + y_{\text{A1},2}) + 0.0321(y_{\text{A2},1} + y_{\text{A2},2} + y_{\text{A2},3}) + 0.025375(y_{\text{A3},1} + y_{\text{A3},2} + y_{\text{A3},3}) \\
+ 0.0625(y_{\text{B1},1} + y_{\text{B1},2}) + 0.111857(y_{\text{B2},1} + y_{\text{B2},3}) + 0.05(y_{\text{B3},1}) + 0.06(y_{\text{B4},1} + y_{\text{B4},2} + y_{\text{B4},3})
\]

All variables are continuous and nonnegative.