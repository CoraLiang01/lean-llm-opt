Let $x_{g,b}$ denote the amount (kg) of raw grade $g \in \{\text{I}, \text{II}, \text{III}\}$ used in wine brand $b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$.

Let $P_b$ be the selling price (CNY/kg) of brand $b$:
- $P_{\text{Red}} = 5.5$
- $P_{\text{Yellow}} = 5$
- $P_{\text{Blue}} = 4.8$

Let $C_g$ be the cost (CNY/kg) of raw grade $g$:
- $C_{\text{I}} = 6$
- $C_{\text{II}} = 4.5$
- $C_{\text{III}} = 3$

Let $S_g$ be the daily supply limit (kg) of raw grade $g$:
- $S_{\text{I}} = 1500$
- $S_{\text{II}} = 2000$
- $S_{\text{III}} = 1000$

Define total production of brand $b$ as $y_b = \sum_{g} x_{g,b}$.

Objective:
\[
\max \left\{ \sum_{b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}} P_b \cdot y_b - \sum_{g \in \{\text{I}, \text{II}, \text{III}\}} C_g \cdot \sum_{b} x_{g,b} \right\}
\]

Subject to:

##### 1. Blending Requirements

- For Red:
  - Proportion of I less than 10%: $\dfrac{x_{\text{I},\text{Red}}}{y_{\text{Red}}} < 0.10$ (if $y_{\text{Red}} > 0$)
  - Proportion of II more than 50%: $\dfrac{x_{\text{II},\text{Red}}}{y_{\text{Red}}} > 0.50$ (if $y_{\text{Red}} > 0$)

- For Yellow:
  - Proportion of III less than 70%: $\dfrac{x_{\text{III},\text{Yellow}}}{y_{\text{Yellow}}} < 0.70$ (if $y_{\text{Yellow}} > 0$)
  - Proportion of I more than 20%: $\dfrac{x_{\text{I},\text{Yellow}}}{y_{\text{Yellow}}} > 0.20$ (if $y_{\text{Yellow}} > 0$)

- For Blue:
  - Proportion of I less than 50%: $\dfrac{x_{\text{I},\text{Blue}}}{y_{\text{Blue}}} < 0.50$ (if $y_{\text{Blue}} > 0$)
  - Proportion of II more than 10%: $\dfrac{x_{\text{II},\text{Blue}}}{y_{\text{Blue}}} > 0.10$ (if $y_{\text{Blue}} > 0$)

These can be written as:
\[
\begin{align*}
& x_{\text{I},\text{Red}} < 0.10\, y_{\text{Red}} \\
& x_{\text{II},\text{Red}} > 0.50\, y_{\text{Red}} \\
& x_{\text{III},\text{Yellow}} < 0.70\, y_{\text{Yellow}} \\
& x_{\text{I},\text{Yellow}} > 0.20\, y_{\text{Yellow}} \\
& x_{\text{I},\text{Blue}} < 0.50\, y_{\text{Blue}} \\
& x_{\text{II},\text{Blue}} > 0.10\, y_{\text{Blue}}
\end{align*}
\]
(If $y_b = 0$, these constraints are vacuously satisfied.)

##### 2. Raw Material Supply Constraints

\[
\begin{align*}
& x_{\text{I},\text{Red}} + x_{\text{I},\text{Yellow}} + x_{\text{I},\text{Blue}} \leq 1500 \\
& x_{\text{II},\text{Red}} + x_{\text{II},\text{Yellow}} + x_{\text{II},\text{Blue}} \leq 2000 \\
& x_{\text{III},\text{Red}} + x_{\text{III},\text{Yellow}} + x_{\text{III},\text{Blue}} \leq 1000
\end{align*}
\]

##### 3. Minimum Production Constraint

\[
y_{\text{Red}} \geq 2000
\]

##### 4. Nonnegativity

\[
x_{g,b} \geq 0 \quad \forall g \in \{\text{I}, \text{II}, \text{III}\},\ b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}
\]

##### 5. Definition of $y_b$

\[
y_b = x_{\text{I},b} + x_{\text{II},b} + x_{\text{III},b} \quad \forall b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}
\]

---

**All coefficients, identifiers, and constraints are as retrieved and in original order.**