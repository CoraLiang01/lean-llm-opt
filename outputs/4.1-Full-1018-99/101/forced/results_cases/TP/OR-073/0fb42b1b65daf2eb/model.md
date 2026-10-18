Let
- $x_1$: quantity of Product I produced (continuous, $\geq 0$)
- $x_2$: quantity of Product II produced (continuous, $\geq 0$)
- $x_3$: quantity of Product III produced (continuous, $\geq 0$)

Let $y_{A1,1}$: amount of Product I processed on A1 (continuous, $\geq 0$), etc.

Define all variables for each feasible (equipment, product) pair, as allowed by the process and data.

#### Parameters (from 43.csv, source order)

- Equipment and their available operating times and full-load costs:
    - A1: 6000 hours, 300 yuan
    - A2: 10000 hours, 321 yuan
    - A3: 8000 hours, 203 yuan
    - B1: 4000 hours, 250 yuan
    - B2: 7000 hours, 783 yuan
    - B3: 4000 hours, 200 yuan
    - B4: 5000 hours, 300 yuan

- Processing times (hours/unit) for each (equipment, product) pair (blank means not allowed):

| Equipment | Product I | Product II | Product III |
|-----------|-----------|------------|-------------|
| A1        | 5         | 10         |             |
| A2        | 7         | 9          | 12          |
| A3        | 6         | 11         | 2           |
| B1        | 6         | 8          |             |
| B2        | 4         |            | 11          |
| B3        | 7         |            |             |
| B4        | 3         | 5          | 8           |

- Raw material cost (yuan/unit): Product I: 0.25, Product II: 0.35, Product III: 0.5
- Unit price (yuan/unit): Product I: 1.25, Product II: 2, Product III: 2.8

#### Decision Variables

Let $y_{e,p}$: amount of product $p$ processed on equipment $e$ (continuous, $\geq 0$), only for allowed $(e,p)$ pairs (i.e., where processing time is given).

Let $x_1, x_2, x_3$ as above.

#### Objective Function

Maximize total profit:
\[
\max \Bigg[
(1.25 - 0.25)x_1 + (2 - 0.35)x_2 + (2.8 - 0.5)x_3
- 300\frac{\sum_{p} t_{A1,p} y_{A1,p}}{6000}
- 321\frac{\sum_{p} t_{A2,p} y_{A2,p}}{10000}
- 203\frac{\sum_{p} t_{A3,p} y_{A3,p}}{8000}
- 250\frac{\sum_{p} t_{B1,p} y_{B1,p}}{4000}
- 783\frac{\sum_{p} t_{B2,p} y_{B2,p}}{7000}
- 200\frac{\sum_{p} t_{B3,p} y_{B3,p}}{4000}
- 300\frac{\sum_{p} t_{B4,p} y_{B4,p}}{5000}
\Bigg]
\]
where $t_{e,p}$ is the processing time per unit of product $p$ on equipment $e$ (from the table above), and the sum over $p$ is only for products allowed on $e$.

#### Constraints

1. **Production-Processing Assignment:**
   - For each product, the sum of its assignments to all allowed equipment for each procedure must equal the production quantity.

   - **Procedure A:**
     - Product I: $y_{A1,1} + y_{A2,1} + y_{A3,1} = x_1$
     - Product II: $y_{A1,2} + y_{A2,2} + y_{A3,2} = x_2$
     - Product III: $y_{A2,3} + y_{A3,3} = x_3$

   - **Procedure B:**
     - Product I: $y_{B1,1} + y_{B2,1} + y_{B3,1} + y_{B4,1} = x_1$
     - Product II: $y_{B1,2} + y_{B4,2} = x_2$
     - Product III: $y_{B2,3} + y_{B4,3} = x_3$

2. **Equipment Time Capacity:**
   - For each equipment $e$, total processing time assigned cannot exceed available time.

   - $5y_{A1,1} + 10y_{A1,2} \leq 6000$
   - $7y_{A2,1} + 9y_{A2,2} + 12y_{A2,3} \leq 10000$
   - $6y_{A3,1} + 11y_{A3,2} + 2y_{A3,3} \leq 8000$
   - $6y_{B1,1} + 8y_{B1,2} \leq 4000$
   - $4y_{B2,1} + 11y_{B2,3} \leq 7000$
   - $7y_{B3,1} \leq 4000$
   - $3y_{B4,1} + 5y_{B4,2} + 8y_{B4,3} \leq 5000$

3. **Non-negativity:**
   - All $x_i \geq 0$, all $y_{e,p} \geq 0$ for allowed $(e,p)$.

#### Variable List (for all allowed assignments):

- $x_1, x_2, x_3 \geq 0$
- $y_{A1,1}, y_{A1,2} \geq 0$
- $y_{A2,1}, y_{A2,2}, y_{A2,3} \geq 0$
- $y_{A3,1}, y_{A3,2}, y_{A3,3} \geq 0$
- $y_{B1,1}, y_{B1,2} \geq 0$
- $y_{B2,1}, y_{B2,3} \geq 0$
- $y_{B3,1} \geq 0$
- $y_{B4,1}, y_{B4,2}, y_{B4,3} \geq 0$

#### Full Model

Maximize:
\[
\begin{align*}
& (1.25 - 0.25)x_1 + (2 - 0.35)x_2 + (2.8 - 0.5)x_3 \\
& - 300\frac{5y_{A1,1} + 10y_{A1,2}}{6000}
- 321\frac{7y_{A2,1} + 9y_{A2,2} + 12y_{A2,3}}{10000}
- 203\frac{6y_{A3,1} + 11y_{A3,2} + 2y_{A3,3}}{8000} \\
& - 250\frac{6y_{B1,1} + 8y_{B1,2}}{4000}
- 783\frac{4y_{B2,1} + 11y_{B2,3}}{7000}
- 200\frac{7y_{B3,1}}{4000}
- 300\frac{3y_{B4,1} + 5y_{B4,2} + 8y_{B4,3}}{5000}
\end{align*}
\]

Subject to:
\[
\begin{align*}
& y_{A1,1} + y_{A2,1} + y_{A3,1} = x_1 \\
& y_{A1,2} + y_{A2,2} + y_{A3,2} = x_2 \\
& y_{A2,3} + y_{A3,3} = x_3 \\
& y_{B1,1} + y_{B2,1} + y_{B3,1} + y_{B4,1} = x_1 \\
& y_{B1,2} + y_{B4,2} = x_2 \\
& y_{B2,3} + y_{B4,3} = x_3 \\
& 5y_{A1,1} + 10y_{A1,2} \leq 6000 \\
& 7y_{A2,1} + 9y_{A2,2} + 12y_{A2,3} \leq 10000 \\
& 6y_{A3,1} + 11y_{A3,2} + 2y_{A3,3} \leq 8000 \\
& 6y_{B1,1} + 8y_{B1,2} \leq 4000 \\
& 4y_{B2,1} + 11y_{B2,3} \leq 7000 \\
& 7y_{B3,1} \leq 4000 \\
& 3y_{B4,1} + 5y_{B4,2} + 8y_{B4,3} \leq 5000 \\
& x_1, x_2, x_3 \geq 0 \\
& y_{A1,1}, y_{A1,2}, y_{A2,1}, y_{A2,2}, y_{A2,3}, y_{A3,1}, y_{A3,2}, y_{A3,3}, \\
& y_{B1,1}, y_{B1,2}, y_{B2,1}, y_{B2,3}, y_{B3,1}, y_{B4,1}, y_{B4,2}, y_{B4,3} \geq 0
\end{align*}
\]

All coefficients and identifiers are as in the retrieved data, in source order.