Let $x_{ij}$ denote the quantity of product $j$ processed on equipment $i$. All $x_{ij} \geq 0$ and continuous.

Define:
- Products: I, II, III
- Equipment: A1, A2, A3 (Procedure A); B1, B2, B3, B4 (Procedure B)
- $c_{ij}$: processing time (hours/unit) of product $j$ on equipment $i$
- $T_i$: available operating time (hours) for equipment $i$
- $F_i$: equipment cost at full load (yuan) for equipment $i$
- $r_j$: raw material cost per unit of product $j$
- $s_j$: selling price per unit of product $j$

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
- $x_{A1,I}$: units of Product I processed on A1
- $x_{A1,II}$: units of Product II processed on A1
- $x_{A2,I}$, $x_{A2,II}$, $x_{A2,III}$
- $x_{A3,I}$, $x_{A3,II}$, $x_{A3,III}$
- $x_{B1,I}$, $x_{B1,II}$
- $x_{B2,I}$, $x_{B2,III}$
- $x_{B3,I}$
- $x_{B4,I}$, $x_{B4,II}$, $x_{B4,III}$

(Only define $x_{ij}$ where a processing time is given.)

#### Objective Function

Maximize total profit = total revenue − total raw material cost − total equipment cost

\[
\max \Bigg[
\underbrace{1.25 \cdot y_I + 2 \cdot y_{II} + 2.8 \cdot y_{III}}_{\text{Total Revenue}}
- \underbrace{0.25 \cdot y_I + 0.35 \cdot y_{II} + 0.5 \cdot y_{III}}_{\text{Raw Material Cost}}
- \sum_{i} F_i \cdot \frac{\text{Total time used on } i}{T_i}
\Bigg]
\]

where:
- $y_I$ = total units of Product I produced (must be the same after A and B)
- $y_{II}$ = total units of Product II produced (same after A and B)
- $y_{III}$ = total units of Product III produced (same after A and B)

#### Constraints

##### 1. Equipment Time Constraints

For each equipment $i$:

\[
\sum_{j} c_{ij} x_{ij} \leq T_i
\]

where $c_{ij}$ is the processing time for product $j$ on equipment $i$ (from table), and $x_{ij}$ is defined only if $c_{ij}$ is given.

##### 2. Flow Balance Constraints

Each product must be processed by one or more A equipment, then by one or more B equipment, with the same total quantity:

- For Product I:
    - After A: $y_I^A = x_{A1,I} + x_{A2,I} + x_{A3,I}$
    - After B: $y_I^B = x_{B1,I} + x_{B2,I} + x_{B3,I} + x_{B4,I}$
    - $y_I^A = y_I^B = y_I$

- For Product II:
    - After A: $y_{II}^A = x_{A1,II} + x_{A2,II} + x_{A3,II}$
    - After B: $y_{II}^B = x_{B1,II} + x_{B4,II}$
    - $y_{II}^A = y_{II}^B = y_{II}$

- For Product III:
    - After A: $y_{III}^A = x_{A2,III} + x_{A3,III}$
    - After B: $y_{III}^B = x_{B2,III} + x_{B4,III}$
    - $y_{III}^A = y_{III}^B = y_{III}$

##### 3. Processing Restrictions

- Product I: can be processed on any A equipment (A1, A2, A3) and any B equipment (B1, B2, B3, B4)
- Product II: can be processed on any A equipment (A1, A2, A3), but only B1 and B4 for B
- Product III: only A2 and A3 for A, only B2 and B4 for B

##### 4. Nonnegativity

All $x_{ij} \geq 0$ and continuous.

---

#### Complete Model

Let the variables be as above.

Maximize:
\[
\begin{align*}
\max \Bigg\{ & [1.25(y_I) + 2(y_{II}) + 2.8(y_{III})] \\
& - [0.25(y_I) + 0.35(y_{II}) + 0.5(y_{III})] \\
& - \Bigg[
300 \cdot \frac{5x_{A1,I} + 10x_{A1,II}}{6000}
+ 321 \cdot \frac{7x_{A2,I} + 9x_{A2,II} + 12x_{A2,III}}{10000}
+ 203 \cdot \frac{6x_{A3,I} + 11x_{A3,II} + 2x_{A3,III}}{8000} \\
& \quad + 250 \cdot \frac{6x_{B1,I} + 8x_{B1,II}}{4000}
+ 783 \cdot \frac{4x_{B2,I} + 11x_{B2,III}}{7000}
+ 200 \cdot \frac{7x_{B3,I}}{4000}
+ 300 \cdot \frac{3x_{B4,I} + 5x_{B4,II} + 8x_{B4,III}}{5000}
\Bigg]
\Bigg\}
\end{align*}
\]

Subject to:

\[
\begin{align*}
& 5x_{A1,I} + 10x_{A1,II} \leq 6000 \\
& 7x_{A2,I} + 9x_{A2,II} + 12x_{A2,III} \leq 10000 \\
& 6x_{A3,I} + 11x_{A3,II} + 2x_{A3,III} \leq 8000 \\
& 6x_{B1,I} + 8x_{B1,II} \leq 4000 \\
& 4x_{B2,I} + 11x_{B2,III} \leq 7000 \\
& 7x_{B3,I} \leq 4000 \\
& 3x_{B4,I} + 5x_{B4,II} + 8x_{B4,III} \leq 5000 \\
\\
& x_{A1,I} + x_{A2,I} + x_{A3,I} = x_{B1,I} + x_{B2,I} + x_{B3,I} + x_{B4,I} = y_I \\
& x_{A1,II} + x_{A2,II} + x_{A3,II} = x_{B1,II} + x_{B4,II} = y_{II} \\
& x_{A2,III} + x_{A3,III} = x_{B2,III} + x_{B4,III} = y_{III} \\
\\
& x_{ij} \geq 0 \quad \text{for all defined } x_{ij}
\end{align*}
\]

where all $x_{ij}$ are continuous and nonnegative, and only defined for the combinations with a processing time in the data above.