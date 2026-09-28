Let $x_{ij}$ denote the quantity of product $j$ processed on equipment $i$. All $x_{ij} \geq 0$ and continuous.

Define:
- Products: I, II, III
- Equipment for procedure A: A1, A2, A3
- Equipment for procedure B: B1, B2, B3, B4

Let $c_{ij}$ be the processing time (minutes/unit) of product $j$ on equipment $i$ (blank means not allowed).
Let $T_i$ be the available operating time for equipment $i$.
Let $F_i$ be the equipment cost at full load for equipment $i$.
Let $r_j$ be the raw material cost per unit for product $j$.
Let $s_j$ be the selling price per unit for product $j$.

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

Raw material cost (yuan/unit): I: 0.25, II: 0.35, III: 0.5

Unit price (yuan/unit): I: 1.25, II: 2, III: 2.8

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
\sum_{j \in \{I,II,III\}} s_j Q_j
- \sum_{j \in \{I,II,III\}} r_j Q_j
- \sum_{i} F_i \cdot y_i
\Bigg]
\]

Where:
- $Q_j$ is the total output of product $j$ (see below).
- $y_i$ is a binary variable: $y_i = 1$ if equipment $i$ is used at all, $0$ otherwise (if fixed costs are only incurred at full load, otherwise omit $y_i$ and use $F_i$ as variable cost per unit time if appropriate).

But since the data says "equipment cost at full load", and no per-unit cost is given, we assume $F_i$ is incurred if equipment $i$ is used at all and at full load. If not, and the cost is proportional to usage, then:

\[
\text{Equipment cost for } i = F_i \cdot \frac{\text{total time used on } i}{T_i}
\]

So, for each equipment $i$:

\[
\text{Total time used on } i = \sum_{j} c_{ij} x_{ij}
\]

So, total equipment cost:

\[
\sum_{i} F_i \cdot \frac{\sum_{j} c_{ij} x_{ij}}{T_i}
\]

Thus, the objective is:

\[
\max \Bigg[
\sum_{j \in \{I,II,III\}} s_j Q_j
- \sum_{j \in \{I,II,III\}} r_j Q_j
- \sum_{i} F_i \cdot \frac{\sum_{j} c_{ij} x_{ij}}{T_i}
\Bigg]
\]

#### Constraints

1. **Production flow for each product:**

Each product must be processed by one A equipment and one B equipment, according to allowed assignments.

Let $Q_j$ be the total output of product $j$.

- For Product I:
    - $Q_I = x_{A1,I} + x_{A2,I} + x_{A3,I} = x_{B1,I} + x_{B2,I} + x_{B3,I} + x_{B4,I}$
- For Product II:
    - $Q_{II} = x_{A1,II} + x_{A2,II} + x_{A3,II} = x_{B1,II} + x_{B4,II}$
- For Product III:
    - $Q_{III} = x_{A2,III} + x_{A3,III} = x_{B2,III} + x_{B4,III}$

2. **Equipment time constraints:**

For each equipment, total processing time cannot exceed available time.

- A1: $5 x_{A1,I} + 10 x_{A1,II} \leq 6000$
- A2: $7 x_{A2,I} + 9 x_{A2,II} + 12 x_{A2,III} \leq 10000$
- A3: $6 x_{A3,I} + 11 x_{A3,II} + 2 x_{A3,III} \leq 8000$
- B1: $6 x_{B1,I} + 8 x_{B1,II} \leq 4000$
- B2: $4 x_{B2,I} + 11 x_{B2,III} \leq 7000$
- B3: $7 x_{B3,I} \leq 4000$
- B4: $3 x_{B4,I} + 5 x_{B4,II} + 8 x_{B4,III} \leq 5000$

3. **Assignment constraints (from process-product-equipment compatibility):**

- $x_{A1,III} = 0$
- $x_{A2,III} \geq 0$
- $x_{A3,III} \geq 0$
- $x_{B1,III} = 0$
- $x_{B2,II} = 0$
- $x_{B2,III} \geq 0$
- $x_{B3,II} = x_{B3,III} = 0$
- $x_{B4,II} \geq 0$, $x_{B4,III} \geq 0$

4. **Nonnegativity:**

All $x_{ij} \geq 0$ and continuous.

#### Complete Model

Let $x_{A1,I}, x_{A1,II}, x_{A2,I}, x_{A2,II}, x_{A2,III}, x_{A3,I}, x_{A3,II}, x_{A3,III}, x_{B1,I}, x_{B1,II}, x_{B2,I}, x_{B2,III}, x_{B3,I}, x_{B4,I}, x_{B4,II}, x_{B4,III} \geq 0$

Maximize:
\[
1.25 Q_I + 2 Q_{II} + 2.8 Q_{III}
- 0.25 Q_I - 0.35 Q_{II} - 0.5 Q_{III}
- \left[
300 \frac{5 x_{A1,I} + 10 x_{A1,II}}{6000}
+ 321 \frac{7 x_{A2,I} + 9 x_{A2,II} + 12 x_{A2,III}}{10000}
+ 203 \frac{6 x_{A3,I} + 11 x_{A3,II} + 2 x_{A3,III}}{8000}
+ 250 \frac{6 x_{B1,I} + 8 x_{B1,II}}{4000}
+ 783 \frac{4 x_{B2,I} + 11 x_{B2,III}}{7000}
+ 200 \frac{7 x_{B3,I}}{4000}
+ 300 \frac{3 x_{B4,I} + 5 x_{B4,II} + 8 x_{B4,III}}{5000}
\right]
\]

Subject to:

\[
\begin{align*}
& Q_I = x_{A1,I} + x_{A2,I} + x_{A3,I} = x_{B1,I} + x_{B2,I} + x_{B3,I} + x_{B4,I} \\
& Q_{II} = x_{A1,II} + x_{A2,II} + x_{A3,II} = x_{B1,II} + x_{B4,II} \\
& Q_{III} = x_{A2,III} + x_{A3,III} = x_{B2,III} + x_{B4,III} \\
& 5 x_{A1,I} + 10 x_{A1,II} \leq 6000 \\
& 7 x_{A2,I} + 9 x_{A2,II} + 12 x_{A2,III} \leq 10000 \\
& 6 x_{A3,I} + 11 x_{A3,II} + 2 x_{A3,III} \leq 8000 \\
& 6 x_{B1,I} + 8 x_{B1,II} \leq 4000 \\
& 4 x_{B2,I} + 11 x_{B2,III} \leq 7000 \\
& 7 x_{B3,I} \leq 4000 \\
& 3 x_{B4,I} + 5 x_{B4,II} + 8 x_{B4,III} \leq 5000 \\
& x_{A1,III} = 0 \\
& x_{B1,III} = 0 \\
& x_{B2,II} = 0 \\
& x_{B3,II} = 0 \\
& x_{B3,III} = 0 \\
& x_{ij} \geq 0 \text{ and continuous for all defined variables}
\end{align*}
\]

Where:
- $x_{A1,I}, x_{A1,II}, x_{A2,I}, x_{A2,II}, x_{A2,III}, x_{A3,I}, x_{A3,II}, x_{A3,III}, x_{B1,I}, x_{B1,II}, x_{B2,I}, x_{B2,III}, x_{B3,I}, x_{B4,I}, x_{B4,II}, x_{B4,III}$ are the decision variables.

All coefficients and identifiers are as in the retrieved data.