Let
- $x_1$: quantity of Product I produced (continuous, $\geq 0$)
- $x_2$: quantity of Product II produced (continuous, $\geq 0$)
- $x_3$: quantity of Product III produced (continuous, $\geq 0$)

Let $y_{A1,I}$, $y_{A1,II}$, $y_{A1,III}$: amount of Product I, II, III processed on A1 (continuous, $\geq 0$), and similarly for $y_{A2,*}$, $y_{A3,*}$, $y_{B1,*}$, $y_{B2,*}$, $y_{B3,*}$, $y_{B4,*}$, as allowed by process-product-equipment compatibility.

#### Parameters (from 43.csv, in source order)

- Equipment types for procedure A: A1, A2, A3
- Equipment types for procedure B: B1, B2, B3, B4

Processing times (minutes per unit):

| Equipment | Product I | Product II | Product III | Available Equipment Operating Time | Equipment Cost at Full Load (yuan) |
|-----------|-----------|------------|-------------|-----------------------------------|------------------------------------|
| A1        | 5         | 10         |             | 6000                              | 300                                |
| A2        | 7         | 9          | 12          | 10000                             | 321                                |
| A3        | 6         | 11         | 2           | 8000                              | 203                                |
| B1        | 6         | 8          |             | 4000                              | 250                                |
| B2        | 4         |            | 11          | 7000                              | 783                                |
| B3        | 7         |            |             | 4000                              | 200                                |
| B4        | 3         | 5          | 8           | 5000                              | 300                                |

Raw material cost (yuan/unit): Product I: 0.25, Product II: 0.35, Product III: 0.5

Unit price (yuan/unit): Product I: 1.25, Product II: 2, Product III: 2.8

#### Product-equipment compatibility (from user description):

- Product I: A1, A2, A3; B1, B2, B3, B4
- Product II: A1, A2, A3; B1, B4
- Product III: A2, A3; B2, B4

#### Decision Variables

Let $y_{A1,I}$ = units of Product I processed on A1  
Let $y_{A1,II}$ = units of Product II processed on A1  
Let $y_{A2,I}$, $y_{A2,II}$, $y_{A2,III}$ = units of Product I, II, III processed on A2  
Let $y_{A3,I}$, $y_{A3,II}$, $y_{A3,III}$ = units of Product I, II, III processed on A3  
Let $y_{B1,I}$, $y_{B1,II}$ = units of Product I, II processed on B1  
Let $y_{B2,I}$, $y_{B2,III}$ = units of Product I, III processed on B2  
Let $y_{B3,I}$ = units of Product I processed on B3  
Let $y_{B4,I}$, $y_{B4,II}$, $y_{B4,III}$ = units of Product I, II, III processed on B4

All variables $\geq 0$ and continuous.

#### Model

Maximize total profit:

\[
\begin{align*}
\max\quad & \text{Total Revenue} - \text{Raw Material Cost} - \text{Equipment Cost} \\
= & [1.25 x_1 + 2 x_2 + 2.8 x_3] \\
  & - [0.25 x_1 + 0.35 x_2 + 0.5 x_3] \\
  & - \left[300 \frac{t_{A1}}{6000} + 321 \frac{t_{A2}}{10000} + 203 \frac{t_{A3}}{8000} + 250 \frac{t_{B1}}{4000} + 783 \frac{t_{B2}}{7000} + 200 \frac{t_{B3}}{4000} + 300 \frac{t_{B4}}{5000}\right]
\end{align*}
\]

where $t_{A1}$, $t_{A2}$, $t_{A3}$, $t_{B1}$, $t_{B2}$, $t_{B3}$, $t_{B4}$ are the total minutes used on each equipment, defined below.

#### Linking constraints (product must be processed in both A and B):

\[
\begin{align*}
x_1 &= y_{A1,I} + y_{A2,I} + y_{A3,I} = y_{B1,I} + y_{B2,I} + y_{B3,I} + y_{B4,I} \\
x_2 &= y_{A1,II} + y_{A2,II} + y_{A3,II} = y_{B1,II} + y_{B4,II} \\
x_3 &= y_{A2,III} + y_{A3,III} = y_{B2,III} + y_{B4,III}
\end{align*}
\]

#### Equipment time usage

\[
\begin{align*}
t_{A1} &= 5 y_{A1,I} + 10 y_{A1,II} \\
t_{A2} &= 7 y_{A2,I} + 9 y_{A2,II} + 12 y_{A2,III} \\
t_{A3} &= 6 y_{A3,I} + 11 y_{A3,II} + 2 y_{A3,III} \\
t_{B1} &= 6 y_{B1,I} + 8 y_{B1,II} \\
t_{B2} &= 4 y_{B2,I} + 11 y_{B2,III} \\
t_{B3} &= 7 y_{B3,I} \\
t_{B4} &= 3 y_{B4,I} + 5 y_{B4,II} + 8 y_{B4,III}
\end{align*}
\]

#### Equipment time capacity constraints

\[
\begin{align*}
t_{A1} &\leq 6000 \\
t_{A2} &\leq 10000 \\
t_{A3} &\leq 8000 \\
t_{B1} &\leq 4000 \\
t_{B2} &\leq 7000 \\
t_{B3} &\leq 4000 \\
t_{B4} &\leq 5000
\end{align*}
\]

#### Variable domains

All $x_1, x_2, x_3, y_{A1,I}, y_{A1,II}, y_{A2,I}, y_{A2,II}, y_{A2,III}, y_{A3,I}, y_{A3,II}, y_{A3,III}, y_{B1,I}, y_{B1,II}, y_{B2,I}, y_{B2,III}, y_{B3,I}, y_{B4,I}, y_{B4,II}, y_{B4,III} \geq 0$ and continuous.

#### Complete Model (all coefficients and identifiers preserved)

\[
\begin{align*}
\max\quad & [1.25 x_1 + 2 x_2 + 2.8 x_3] - [0.25 x_1 + 0.35 x_2 + 0.5 x_3] \\
& - \left[300 \frac{t_{A1}}{6000} + 321 \frac{t_{A2}}{10000} + 203 \frac{t_{A3}}{8000} + 250 \frac{t_{B1}}{4000} + 783 \frac{t_{B2}}{7000} + 200 \frac{t_{B3}}{4000} + 300 \frac{t_{B4}}{5000}\right] \\
\text{s.t.}\quad
& x_1 = y_{A1,I} + y_{A2,I} + y_{A3,I} \\
& x_1 = y_{B1,I} + y_{B2,I} + y_{B3,I} + y_{B4,I} \\
& x_2 = y_{A1,II} + y_{A2,II} + y_{A3,II} \\
& x_2 = y_{B1,II} + y_{B4,II} \\
& x_3 = y_{A2,III} + y_{A3,III} \\
& x_3 = y_{B2,III} + y_{B4,III} \\
& 5 y_{A1,I} + 10 y_{A1,II} \leq 6000 \\
& 7 y_{A2,I} + 9 y_{A2,II} + 12 y_{A2,III} \leq 10000 \\
& 6 y_{A3,I} + 11 y_{A3,II} + 2 y_{A3,III} \leq 8000 \\
& 6 y_{B1,I} + 8 y_{B1,II} \leq 4000 \\
& 4 y_{B2,I} + 11 y_{B2,III} \leq 7000 \\
& 7 y_{B3,I} \leq 4000 \\
& 3 y_{B4,I} + 5 y_{B4,II} + 8 y_{B4,III} \leq 5000 \\
& x_1, x_2, x_3 \geq 0 \\
& y_{A1,I}, y_{A1,II}, y_{A2,I}, y_{A2,II}, y_{A2,III}, y_{A3,I}, y_{A3,II}, y_{A3,III}, y_{B1,I}, y_{B1,II}, y_{B2,I}, y_{B2,III}, y_{B3,I}, y_{B4,I}, y_{B4,II}, y_{B4,III} \geq 0
\end{align*}
\]

All coefficients and identifiers are preserved as in the source data.