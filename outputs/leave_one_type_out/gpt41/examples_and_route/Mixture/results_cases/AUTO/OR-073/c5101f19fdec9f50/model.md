Let $x_{ij}$ denote the quantity of product $j$ processed on equipment $i$. All $x_{ij} \geq 0$ and continuous.

Define:
- Products: I, II, III
- Equipment: A1, A2, A3 (Procedure A); B1, B2, B3, B4 (Procedure B)
- $c_{ij}$: processing time per unit of product $j$ on equipment $i$ (from table)
- $T_i$: available operating time for equipment $i$
- $F_i$: equipment cost at full load for equipment $i$
- $r_j$: raw material cost per unit of product $j$
- $p_j$: selling price per unit of product $j$

From the data:

| Equipment | Product I | Product II | Product III | Available Equipment Operating Time | Equipment Cost at Full Load (yuan) |
|-----------|-----------|------------|-------------|-----------------------------------|------------------------------------|
| A1        | 5         | 10         |             | 6000                              | 300                                |
| A2        | 7         | 9          | 12          | 10000                             | 321                                |
| A3        | 6         | 11         | 2           | 8000                              | 203                                |
| B1        | 6         | 8          |             | 4000                              | 250                                |
| B2        | 4         |            | 11          | 7000                              | 783                                |
| B3        | 7         |            |             | 4000                              | 200                                |
| B4        | 3         | 5          | 8           | 5000                              | 300                                |

Raw Material Cost (yuan/unit): I: 0.25, II: 0.35, III: 0.5

Unit Price (yuan/unit): I: 1.25, II: 2, III: 2.8

#### Decision Variables

Let:
- $x_{A1,I}$, $x_{A1,II}$: units of I, II processed on A1
- $x_{A2,I}$, $x_{A2,II}$, $x_{A2,III}$: units of I, II, III processed on A2
- $x_{A3,I}$, $x_{A3,II}$, $x_{A3,III}$: units of I, II, III processed on A3
- $x_{B1,I}$, $x_{B1,II}$: units of I, II processed on B1
- $x_{B2,I}$, $x_{B2,III}$: units of I, III processed on B2
- $x_{B3,I}$: units of I processed on B3
- $x_{B4,I}$, $x_{B4,II}$, $x_{B4,III}$: units of I, II, III processed on B4

#### Objective Function

Maximize total profit:

\[
\max \Bigg[
\underbrace{1.25}_{p_I} \cdot y_I + \underbrace{2}_{p_{II}} \cdot y_{II} + \underbrace{2.8}_{p_{III}} \cdot y_{III}
- \underbrace{0.25}_{r_I} \cdot y_I - \underbrace{0.35}_{r_{II}} \cdot y_{II} - \underbrace{0.5}_{r_{III}} \cdot y_{III}
- \sum_{i} F_i \cdot \frac{u_i}{T_i}
\Bigg]
\]

where:
- $y_I$, $y_{II}$, $y_{III}$: total output of products I, II, III (see below)
- $u_i$: total time used on equipment $i$ (see below)
- $F_i$: equipment cost at full load for equipment $i$
- $T_i$: available operating time for equipment $i$

#### Constraints

##### 1. Product Flow Constraints

Each product must be processed by one A equipment and one B equipment, subject to process routes:

- Product I:
    - $y_I = x_{A1,I} + x_{A2,I} + x_{A3,I} = x_{B1,I} + x_{B2,I} + x_{B3,I} + x_{B4,I}$
- Product II:
    - $y_{II} = x_{A1,II} + x_{A2,II} + x_{A3,II} = x_{B1,II} + x_{B4,II}$
- Product III:
    - $y_{III} = x_{A2,III} + x_{A3,III} = x_{B2,III} + x_{B4,III}$

##### 2. Equipment Time Constraints

For each equipment, total processing time cannot exceed available time:

- A1: $5x_{A1,I} + 10x_{A1,II} \leq 6000$
- A2: $7x_{A2,I} + 9x_{A2,II} + 12x_{A2,III} \leq 10000$
- A3: $6x_{A3,I} + 11x_{A3,II} + 2x_{A3,III} \leq 8000$
- B1: $6x_{B1,I} + 8x_{B1,II} \leq 4000$
- B2: $4x_{B2,I} + 11x_{B2,III} \leq 7000$
- B3: $7x_{B3,I} \leq 4000$
- B4: $3x_{B4,I} + 5x_{B4,II} + 8x_{B4,III} \leq 5000$

##### 3. Equipment Cost Calculation

For each equipment $i$, let $u_i$ be the total time used:

- $u_{A1} = 5x_{A1,I} + 10x_{A1,II}$
- $u_{A2} = 7x_{A2,I} + 9x_{A2,II} + 12x_{A2,III}$
- $u_{A3} = 6x_{A3,I} + 11x_{A3,II} + 2x_{A3,III}$
- $u_{B1} = 6x_{B1,I} + 8x_{B1,II}$
- $u_{B2} = 4x_{B2,I} + 11x_{B2,III}$
- $u_{B3} = 7x_{B3,I}$
- $u_{B4} = 3x_{B4,I} + 5x_{B4,II} + 8x_{B4,III}$

The cost for each equipment is proportional to its usage: $F_i \cdot \frac{u_i}{T_i}$

##### 4. Feasibility Constraints (Process Routing)

Variables not allowed by process routes are set to zero:

- $x_{A1,III} = 0$
- $x_{B1,III} = 0$
- $x_{A2,II}$, $x_{A2,III}$, etc., only as allowed by the table (see above)
- $x_{B2,II} = 0$, $x_{B3,II} = 0$, $x_{B3,III} = 0$, $x_{B4,II}$, $x_{B4,III}$ as allowed

##### 5. Nonnegativity

All $x_{ij} \geq 0$ and continuous.

---

#### Complete Model

Maximize:
\[
\begin{align*}
\max \Bigg\{ &
[1.25 \cdot y_I + 2 \cdot y_{II} + 2.8 \cdot y_{III}]
- [0.25 \cdot y_I + 0.35 \cdot y_{II} + 0.5 \cdot y_{III}] \\
& - \Big[
300 \cdot \frac{u_{A1}}{6000}
+ 321 \cdot \frac{u_{A2}}{10000}
+ 203 \cdot \frac{u_{A3}}{8000}
+ 250 \cdot \frac{u_{B1}}{4000}
+ 783 \cdot \frac{u_{B2}}{7000}
+ 200 \cdot \frac{u_{B3}}{4000}
+ 300 \cdot \frac{u_{B4}}{5000}
\Big]
\Bigg\}
\end{align*}
\]

Subject to:

\[
\begin{align*}
& x_{A1,I} + x_{A2,I} + x_{A3,I} = x_{B1,I} + x_{B2,I} + x_{B3,I} + x_{B4,I} = y_I \\
& x_{A1,II} + x_{A2,II} + x_{A3,II} = x_{B1,II} + x_{B4,II} = y_{II} \\
& x_{A2,III} + x_{A3,III} = x_{B2,III} + x_{B4,III} = y_{III} \\
& 5x_{A1,I} + 10x_{A1,II} \leq 6000 \\
& 7x_{A2,I} + 9x_{A2,II} + 12x_{A2,III} \leq 10000 \\
& 6x_{A3,I} + 11x_{A3,II} + 2x_{A3,III} \leq 8000 \\
& 6x_{B1,I} + 8x_{B1,II} \leq 4000 \\
& 4x_{B2,I} + 11x_{B2,III} \leq 7000 \\
& 7x_{B3,I} \leq 4000 \\
& 3x_{B4,I} + 5x_{B4,II} + 8x_{B4,III} \leq 5000 \\
& x_{A1,III} = 0,\ x_{B1,III} = 0,\ x_{B2,II} = 0,\ x_{B3,II} = 0,\ x_{B3,III} = 0 \\
& x_{ij} \geq 0 \quad \text{for all allowed } i,j
\end{align*}
\]

where:

\[
\begin{align*}
u_{A1} &= 5x_{A1,I} + 10x_{A1,II} \\
u_{A2} &= 7x_{A2,I} + 9x_{A2,II} + 12x_{A2,III} \\
u_{A3} &= 6x_{A3,I} + 11x_{A3,II} + 2x_{A3,III} \\
u_{B1} &= 6x_{B1,I} + 8x_{B1,II} \\
u_{B2} &= 4x_{B2,I} + 11x_{B2,III} \\
u_{B3} &= 7x_{B3,I} \\
u_{B4} &= 3x_{B4,I} + 5x_{B4,II} + 8x_{B4,III}
\end{align*}
\]

All variables $x_{ij} \geq 0$ and continuous.