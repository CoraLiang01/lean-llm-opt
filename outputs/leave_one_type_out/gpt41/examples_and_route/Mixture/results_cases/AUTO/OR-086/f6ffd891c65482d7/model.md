Let $x_{g,b}$ denote the kilograms of raw grade $g$ used in brand $b$, where $g \in \{\text{I}, \text{II}, \text{III}\}$ and $b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$.

Let $S_g$ be the daily supply limit (kg) and $C_g$ the unit cost (CNY/kg) for grade $g$ (from 30-1.csv):

\[
\begin{align*}
S_{\text{I}} &= 1500, \quad C_{\text{I}} = 6 \\
S_{\text{II}} &= 2000, \quad C_{\text{II}} = 4.5 \\
S_{\text{III}} &= 1000, \quad C_{\text{III}} = 3 \\
\end{align*}
\]

Let $P_b$ be the selling price (CNY/kg) for brand $b$ (from 30-2.csv):

\[
\begin{align*}
P_{\text{Red}} &= 5.5 \\
P_{\text{Yellow}} &= 5 \\
P_{\text{Blue}} &= 4.8 \\
\end{align*}
\]

Define total production of each brand:
\[
y_b = \sum_{g} x_{g,b} \qquad \forall b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}
\]

Objective:
\[
\max \left[ \sum_{b} P_b y_b - \sum_{g} C_g \left( \sum_{b} x_{g,b} \right) \right]
\]

Subject to:

Blending Requirements (from 30-2.csv):

- Red: I less than 10%, II more than 50%
  \[
  \frac{x_{\text{I},\text{Red}}}{y_{\text{Red}}} \leq 0.10 \qquad \text{if } y_{\text{Red}} > 0
  \]
  \[
  \frac{x_{\text{II},\text{Red}}}{y_{\text{Red}}} \geq 0.50 \qquad \text{if } y_{\text{Red}} > 0
  \]

- Yellow: III less than 70%, I more than 20%
  \[
  \frac{x_{\text{III},\text{Yellow}}}{y_{\text{Yellow}}} \leq 0.70 \qquad \text{if } y_{\text{Yellow}} > 0
  \]
  \[
  \frac{x_{\text{I},\text{Yellow}}}{y_{\text{Yellow}}} \geq 0.20 \qquad \text{if } y_{\text{Yellow}} > 0
  \]

- Blue: I less than 50%, II more than 10%
  \[
  \frac{x_{\text{I},\text{Blue}}}{y_{\text{Blue}}} \leq 0.50 \qquad \text{if } y_{\text{Blue}} > 0
  \]
  \[
  \frac{x_{\text{II},\text{Blue}}}{y_{\text{Blue}}} \geq 0.10 \qquad \text{if } y_{\text{Blue}} > 0
  \]

Raw Material Supply Constraints:
\[
\sum_{b} x_{g,b} \leq S_g \qquad \forall g \in \{\text{I}, \text{II}, \text{III}\}
\]

Minimum Production Constraint:
\[
y_{\text{Red}} \geq 2000
\]

Nonnegativity:
\[
x_{g,b} \geq 0 \qquad \forall g, b
\]

Summary of variables and parameters:
- $x_{g,b}$: kg of grade $g$ used in brand $b$ (continuous, $\geq 0$)
- $y_b$: total kg of brand $b$ produced ($y_b = \sum_g x_{g,b}$)
- $S_g$: daily supply limit for grade $g$ (from 30-1.csv)
- $C_g$: unit cost for grade $g$ (from 30-1.csv)
- $P_b$: selling price for brand $b$ (from 30-2.csv)

All coefficients and constraints are as retrieved from the data.