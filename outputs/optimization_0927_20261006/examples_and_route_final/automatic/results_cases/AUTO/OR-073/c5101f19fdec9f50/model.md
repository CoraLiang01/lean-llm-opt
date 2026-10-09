Let:
- $x_1$ = quantity of Product I produced (continuous, $\geq 0$)
- $x_2$ = quantity of Product II produced (continuous, $\geq 0$)
- $x_3$ = quantity of Product III produced (continuous, $\geq 0$)

Let $y_{ij}$ = amount of product $j$ processed on equipment $i$ (continuous, $\geq 0$), where $i$ indexes equipment and $j$ indexes products.

Define equipment and their roles:
- A1, A2, A3: Procedure A
- B1, B2, B3, B4: Procedure B

Parameters (from 43.csv):

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

#### Decision Variables

- $x_1, x_2, x_3 \geq 0$ (continuous)
- $y_{ij} \geq 0$ (continuous), for all feasible $(i,j)$ pairs as per process/equipment compatibility

#### Objective Function

Maximize total profit = total revenue - total raw material cost - total equipment cost

\[
\max \Bigg[
(1.25)x_1 + (2)x_2 + (2.8)x_3
- (0.25)x_1 - (0.35)x_2 - (0.5)x_3
- \sum_{i} \text{Equipment Cost}_i \cdot \left( \frac{\text{Total time used on } i}{\text{Available Equipment Operating Time}_i} \right)
\Bigg]
\]

where
- $\text{Total time used on } i = \sum_{j} (\text{Processing time per unit of } j \text{ on } i) \cdot y_{ij}$

#### Constraints

##### 1. Flow balance: Each product must be fully processed in both procedures

For each product, the total amount processed in each procedure must equal the production quantity:

- For Product I:
    - Procedure A: $y_{A1,1} + y_{A2,1} + y_{A3,1} = x_1$
    - Procedure B: $y_{B1,1} + y_{B2,1} + y_{B3,1} + y_{B4,1} = x_1$
- For Product II:
    - Procedure A: $y_{A1,2} + y_{A2,2} + y_{A3,2} = x_2$
    - Procedure B: $y_{B1,2} + y_{B4,2} = x_2$
- For Product III:
    - Procedure A: $y_{A2,3} + y_{A3,3} = x_3$
    - Procedure B: $y_{B2,3} + y_{B4,3} = x_3$

##### 2. Equipment time capacity

For each equipment $i$:

\[
\sum_{j} (\text{Processing time per unit of } j \text{ on } i) \cdot y_{ij} \leq \text{Available Equipment Operating Time}_i
\]

Explicitly:

- A1: $5y_{A1,1} + 10y_{A1,2} \leq 6000$
- A2: $7y_{A2,1} + 9y_{A2,2} + 12y_{A2,3} \leq 10000$
- A3: $6y_{A3,1} + 11y_{A3,2} + 2y_{A3,3} \leq 8000$
- B1: $6y_{B1,1} + 8y_{B1,2} \leq 4000$
- B2: $4y_{B2,1} + 11y_{B2,3} \leq 7000$
- B3: $7y_{B3,1} \leq 4000$
- B4: $3y_{B4,1} + 5y_{B4,2} + 8y_{B4,3} \leq 5000$

##### 3. Equipment cost calculation

For each equipment $i$:

\[
\text{Equipment Cost}_i \cdot \left( \frac{\sum_{j} (\text{Processing time per unit of } j \text{ on } i) \cdot y_{ij}}{\text{Available Equipment Operating Time}_i} \right)
\]

##### 4. Feasibility (process/equipment compatibility)

Set $y_{ij} = 0$ for infeasible $(i,j)$ pairs (i.e., where processing time is blank in the table).

##### 5. Nonnegativity

All $x_j \geq 0$, all $y_{ij} \geq 0$.

---

#### Complete Model

Let the variables be:

- $x_1, x_2, x_3 \geq 0$
- $y_{A1,1}, y_{A1,2} \geq 0$
- $y_{A2,1}, y_{A2,2}, y_{A2,3} \geq 0$
- $y_{A3,1}, y_{A3,2}, y_{A3,3} \geq 0$
- $y_{B1,1}, y_{B1,2} \geq 0$
- $y_{B2,1}, y_{B2,3} \geq 0$
- $y_{B3,1} \geq 0$
- $y_{B4,1}, y_{B4,2}, y_{B4,3} \geq 0$

Maximize:

\[
\begin{align*}
\max\ & [1.25x_1 + 2x_2 + 2.8x_3 - 0.25x_1 - 0.35x_2 - 0.5x_3 \\
& - 300 \cdot \frac{5y_{A1,1} + 10y_{A1,2}}{6000}
- 321 \cdot \frac{7y_{A2,1} + 9y_{A2,2} + 12y_{A2,3}}{10000}
- 203 \cdot \frac{6y_{A3,1} + 11y_{A3,2} + 2y_{A3,3}}{8000} \\
& - 250 \cdot \frac{6y_{B1,1} + 8y_{B1,2}}{4000}
- 783 \cdot \frac{4y_{B2,1} + 11y_{B2,3}}{7000}
- 200 \cdot \frac{7y_{B3,1}}{4000}
- 300 \cdot \frac{3y_{B4,1} + 5y_{B4,2} + 8y_{B4,3}}{5000}
]
\end{align*}
\]

Subject to:

\[
\begin{align*}
& y_{A1,1} + y_{A2,1} + y_{A3,1} = x_1 \\
& y_{B1,1} + y_{B2,1} + y_{B3,1} + y_{B4,1} = x_1 \\
& y_{A1,2} + y_{A2,2} + y_{A3,2} = x_2 \\
& y_{B1,2} + y_{B4,2} = x_2 \\
& y_{A2,3} + y_{A3,3} = x_3 \\
& y_{B2,3} + y_{B4,3} = x_3 \\
& 5y_{A1,1} + 10y_{A1,2} \leq 6000 \\
& 7y_{A2,1} + 9y_{A2,2} + 12y_{A2,3} \leq 10000 \\
& 6y_{A3,1} + 11y_{A3,2} + 2y_{A3,3} \leq 8000 \\
& 6y_{B1,1} + 8y_{B1,2} \leq 4000 \\
& 4y_{B2,1} + 11y_{B2,3} \leq 7000 \\
& 7y_{B3,1} \leq 4000 \\
& 3y_{B4,1} + 5y_{B4,2} + 8y_{B4,3} \leq 5000 \\
& x_1, x_2, x_3 \geq 0 \\
& y_{A1,1}, y_{A1,2}, y_{A2,1}, y_{A2,2}, y_{A2,3}, y_{A3,1}, y_{A3,2}, y_{A3,3}, y_{B1,1}, y_{B1,2}, y_{B2,1}, y_{B2,3}, y_{B3,1}, y_{B4,1}, y_{B4,2}, y_{B4,3} \geq 0
\end{align*}
\]

where all $y_{ij}$ not listed above are fixed at zero (i.e., infeasible assignments).

All variables are continuous and nonnegative.