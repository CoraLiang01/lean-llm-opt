Let $x_{i}$ denote the production quantity of product $i$ ($i \in \{\text{I}, \text{II}, \text{III}\}$), all continuous and $\geq 0$.

Let $y_{A1,I}$, $y_{A1,II}$, $y_{A2,I}$, $y_{A2,II}$, $y_{A2,III}$, $y_{B1,I}$, $y_{B1,II}$, $y_{B2,I}$, $y_{B2,III}$, $y_{B3,I}$, $y_{B4,I}$, $y_{B4,II}$, $y_{B4,III}$ denote the amount of each product processed on each eligible equipment (continuous, $\geq 0$), as permitted by the data and eligibility rules.

#### Parameters (from 43.csv, in source order):

- Equipment and their available operating times and full-load costs:
    - A1: 6000 hours, 300 yuan
    - A2: 10000 hours, 321 yuan
    - A3: 8000 hours, 203 yuan
    - B1: 4000 hours, 250 yuan
    - B2: 7000 hours, 783 yuan
    - B3: 4000 hours, 200 yuan
    - B4: 5000 hours, 300 yuan

- Processing times (hours/unit):

    | Equipment | Product I | Product II | Product III |
    |-----------|-----------|------------|-------------|
    | A1        | 5         | 10         |             |
    | A2        | 7         | 9          | 12          |
    | A3        | 6         | 11         | 2           |
    | B1        | 6         | 8          |             |
    | B2        | 4         |            | 11          |
    | B3        | 7         |            |             |
    | B4        | 3         | 5          | 8           |

- Raw material cost (yuan/unit): I: 0.25, II: 0.35, III: 0.5

- Unit price (yuan/unit): I: 1.25, II: 2, III: 2.8

#### Eligibility (from user description):

- Product I: A1, A2, A3 for procedure A; B1, B2, B3, B4 for procedure B
- Product II: A1, A2, A3 for procedure A; B1, B4 for procedure B
- Product III: A2, A3 for procedure A; B2, B4 for procedure B

#### Decision Variables

Let:
- $x_{\text{I}}$, $x_{\text{II}}$, $x_{\text{III}}$: total production quantity of each product (continuous, $\geq 0$)
- $y_{A1,I}$, $y_{A2,I}$, $y_{A3,I}$: amount of product I processed on A1, A2, A3 (procedure A)
- $y_{A1,II}$, $y_{A2,II}$, $y_{A3,II}$: amount of product II processed on A1, A2, A3 (procedure A)
- $y_{A2,III}$, $y_{A3,III}$: amount of product III processed on A2, A3 (procedure A)
- $y_{B1,I}$, $y_{B2,I}$, $y_{B3,I}$, $y_{B4,I}$: amount of product I processed on B1, B2, B3, B4 (procedure B)
- $y_{B1,II}$, $y_{B4,II}$: amount of product II processed on B1, B4 (procedure B)
- $y_{B2,III}$, $y_{B4,III}$: amount of product III processed on B2, B4 (procedure B)

#### Objective Function

Maximize total profit = total revenue - total raw material cost - total equipment cost at full load

\[
\max \Bigg[
\begin{aligned}
&1.25\, x_{\text{I}} + 2\, x_{\text{II}} + 2.8\, x_{\text{III}} \\
&- (0.25\, x_{\text{I}} + 0.35\, x_{\text{II}} + 0.5\, x_{\text{III}}) \\
&- \Bigg(
\frac{300}{6000} \cdot \big(5\, y_{A1,I} + 10\, y_{A1,II}\big)
+ \frac{321}{10000} \cdot \big(7\, y_{A2,I} + 9\, y_{A2,II} + 12\, y_{A2,III}\big) \\
&\quad + \frac{203}{8000} \cdot \big(6\, y_{A3,I} + 11\, y_{A3,II} + 2\, y_{A3,III}\big) \\
&\quad + \frac{250}{4000} \cdot \big(6\, y_{B1,I} + 8\, y_{B1,II}\big)
+ \frac{783}{7000} \cdot \big(4\, y_{B2,I} + 11\, y_{B2,III}\big) \\
&\quad + \frac{200}{4000} \cdot 7\, y_{B3,I}
+ \frac{300}{5000} \cdot \big(3\, y_{B4,I} + 5\, y_{B4,II} + 8\, y_{B4,III}\big)
\Bigg)
\end{aligned}
\Bigg]
\]

#### Constraints

**1. Production assignment (each product's total production must be assigned to eligible equipment for each procedure):**

- Procedure A:
    - $x_{\text{I}} = y_{A1,I} + y_{A2,I} + y_{A3,I}$
    - $x_{\text{II}} = y_{A1,II} + y_{A2,II} + y_{A3,II}$
    - $x_{\text{III}} = y_{A2,III} + y_{A3,III}$

- Procedure B:
    - $x_{\text{I}} = y_{B1,I} + y_{B2,I} + y_{B3,I} + y_{B4,I}$
    - $x_{\text{II}} = y_{B1,II} + y_{B4,II}$
    - $x_{\text{III}} = y_{B2,III} + y_{B4,III}$

**2. Equipment operating time constraints (do not exceed available time):**

- A1: $5\, y_{A1,I} + 10\, y_{A1,II} \leq 6000$
- A2: $7\, y_{A2,I} + 9\, y_{A2,II} + 12\, y_{A2,III} \leq 10000$
- A3: $6\, y_{A3,I} + 11\, y_{A3,II} + 2\, y_{A3,III} \leq 8000$
- B1: $6\, y_{B1,I} + 8\, y_{B1,II} \leq 4000$
- B2: $4\, y_{B2,I} + 11\, y_{B2,III} \leq 7000$
- B3: $7\, y_{B3,I} \leq 4000$
- B4: $3\, y_{B4,I} + 5\, y_{B4,II} + 8\, y_{B4,III} \leq 5000$

**3. Nonnegativity and continuity:**

All variables $x_{\text{I}}, x_{\text{II}}, x_{\text{III}}, y_{A1,I}, y_{A2,I}, y_{A3,I}, y_{A1,II}, y_{A2,II}, y_{A3,II}, y_{A2,III}, y_{A3,III}, y_{B1,I}, y_{B2,I}, y_{B3,I}, y_{B4,I}, y_{B1,II}, y_{B4,II}, y_{B2,III}, y_{B4,III} \geq 0$ and continuous.

---

**All coefficients, identifiers, and constraints are preserved in source order and as given in the data.**