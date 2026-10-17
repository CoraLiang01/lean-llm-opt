Let the variables be:
- $x_{A1,I}$: Amount of Product I processed on A1 (continuous, $\geq 0$)
- $x_{A2,I}$: Amount of Product I processed on A2 (continuous, $\geq 0$)
- $x_{A1,II}$: Amount of Product II processed on A1 (continuous, $\geq 0$)
- $x_{A2,II}$: Amount of Product II processed on A2 (continuous, $\geq 0$)
- $x_{A2,III}$: Amount of Product III processed on A2 (continuous, $\geq 0$)

- $x_{B1,I}$: Amount of Product I processed on B1 (continuous, $\geq 0$)
- $x_{B2,I}$: Amount of Product I processed on B2 (continuous, $\geq 0$)
- $x_{B3,I}$: Amount of Product I processed on B3 (continuous, $\geq 0$)
- $x_{B1,II}$: Amount of Product II processed on B1 (continuous, $\geq 0$)
- $x_{B2,III}$: Amount of Product III processed on B2 (continuous, $\geq 0$)

Let $y_1$, $y_2$, $y_3$ be the total production of Product I, II, III, respectively.

#### Objective Function

Maximize total profit:
\[
\max \Bigg[
(1.25 - 0.25) y_1 + (2 - 0.35) y_2 + (2.8 - 0.5) y_3
- \Bigg(
\frac{300}{6000}(5x_{A1,I} + 10x_{A1,II}) +
\frac{321}{10000}(7x_{A2,I} + 9x_{A2,II} + 12x_{A2,III}) +
\frac{250}{4000}(6x_{B1,I} + 8x_{B1,II}) +
\frac{783}{7000}(4x_{B2,I} + 11x_{B2,III}) +
\frac{200}{4000}(7x_{B3,I})
\Bigg)
\Bigg]
\]

#### Constraints

**1. Product flow constraints (each product must be processed through both A and B):**
\[
\begin{align*}
& y_1 = x_{A1,I} + x_{A2,I} = x_{B1,I} + x_{B2,I} + x_{B3,I} \\
& y_2 = x_{A1,II} + x_{A2,II} = x_{B1,II} \\
& y_3 = x_{A2,III} = x_{B2,III}
\end{align*}
\]

**2. Equipment time constraints:**

- A1: $5x_{A1,I} + 10x_{A1,II} \leq 6000$
- A2: $7x_{A2,I} + 9x_{A2,II} + 12x_{A2,III} \leq 10000$
- B1: $6x_{B1,I} + 8x_{B1,II} \leq 4000$
- B2: $4x_{B2,I} + 11x_{B2,III} \leq 7000$
- B3: $7x_{B3,I} \leq 4000$

**3. Technology eligibility constraints:**

- $x_{A1,III} = 0$ (Product III cannot be processed on A1)
- $x_{B1,III} = 0$, $x_{B3,III} = 0$ (Product III can only be processed on B2)
- $x_{B2,II} = 0$, $x_{B3,II} = 0$ (Product II can only be processed on B1)
- $x_{A1,III} = 0$, $x_{A2,III} \geq 0$ (Product III only on A2)

**4. Nonnegativity:**

All $x_{\cdot,\cdot} \geq 0$, all $y_k \geq 0$.

---

#### Data Used (from 43.csv, in source order):

| Equipment / Cost | Product I | Product II | Product III | Available Equipment Operating Time | Equipment Cost at Full Load (yuan) |
|------------------|-----------|------------|-------------|-----------------------------------|------------------------------------|
| A1               | 5         | 10         |             | 6000                              | 300                                |
| A2               | 7         | 9          | 12          | 10000                             | 321                                |
| B1               | 6         | 8          |             | 4000                              | 250                                |
| B2               | 4         |            | 11          | 7000                              | 783                                |
| B3               | 7         |            |             | 4000                              | 200                                |
| Raw Material Cost (yuan/unit) | 0.25 | 0.35 | 0.5 | | |
| Unit Price (yuan/unit) | 1.25 | 2 | 2.8 | | |

---

**Summary of variables:**

- $x_{A1,I}, x_{A2,I}, x_{A1,II}, x_{A2,II}, x_{A2,III}, x_{B1,I}, x_{B2,I}, x_{B3,I}, x_{B1,II}, x_{B2,III} \geq 0$
- $y_1, y_2, y_3 \geq 0$

**All variables are continuous.**