Let $x_{g,b}$ denote the amount (in kg) of raw grade $g \in \{\text{I}, \text{II}, \text{III}\}$ used in brand $b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$.

Let $p_b$ be the selling price (CNY/kg) of brand $b$:
\[
p_{\text{Red}} = 5.5,\quad p_{\text{Yellow}} = 5,\quad p_{\text{Blue}} = 4.8
\]

Let $c_g$ be the cost (CNY/kg) of grade $g$:
\[
c_{\text{I}} = 6,\quad c_{\text{II}} = 4.5,\quad c_{\text{III}} = 3
\]

Let $S_g$ be the daily supply limit (kg) for grade $g$:
\[
S_{\text{I}} = 1500,\quad S_{\text{II}} = 2000,\quad S_{\text{III}} = 1000
\]

Define $y_b = \sum_{g} x_{g,b}$ as the total production (kg) of brand $b$.

Objective:
\[
\max \left\{ \sum_{b \in \{\text{Red},\text{Yellow},\text{Blue}\}} p_b \cdot y_b - \sum_{g \in \{\text{I},\text{II},\text{III}\}} c_g \cdot \sum_{b} x_{g,b} \right\}
\]

Subject to:

Blending Requirements (from 30-2.csv):

- Red: I less than 10%, II more than 50%
  \[
  \frac{x_{\text{I},\text{Red}}}{y_{\text{Red}}} < 0.10 \qquad \text{(if } y_{\text{Red}} > 0\text{)}
  \]
  \[
  \frac{x_{\text{II},\text{Red}}}{y_{\text{Red}}} > 0.50 \qquad \text{(if } y_{\text{Red}} > 0\text{)}
  \]

- Yellow: III less than 70%, I more than 20%
  \[
  \frac{x_{\text{III},\text{Yellow}}}{y_{\text{Yellow}}} < 0.70 \qquad \text{(if } y_{\text{Yellow}} > 0\text{)}
  \]
  \[
  \frac{x_{\text{I},\text{Yellow}}}{y_{\text{Yellow}}} > 0.20 \qquad \text{(if } y_{\text{Yellow}} > 0\text{)}
  \]

- Blue: I less than 50%, II more than 10%
  \[
  \frac{x_{\text{I},\text{Blue}}}{y_{\text{Blue}}} < 0.50 \qquad \text{(if } y_{\text{Blue}} > 0\text{)}
  \]
  \[
  \frac{x_{\text{II},\text{Blue}}}{y_{\text{Blue}}} > 0.10 \qquad \text{(if } y_{\text{Blue}} > 0\text{)}
  \]

Raw Material Supply Constraints:
\[
\sum_{b} x_{g,b} \leq S_g \qquad \forall g \in \{\text{I},\text{II},\text{III}\}
\]

Minimum Production Constraint:
\[
y_{\text{Red}} \geq 2000
\]

Nonnegativity:
\[
x_{g,b} \geq 0 \qquad \forall g, b
\]

Expanded, the model is:

\[
\max \Bigg\{
5.5 \cdot y_{\text{Red}} + 5 \cdot y_{\text{Yellow}} + 4.8 \cdot y_{\text{Blue}}
- 6 \cdot \left(\sum_{b} x_{\text{I},b}\right)
- 4.5 \cdot \left(\sum_{b} x_{\text{II},b}\right)
- 3 \cdot \left(\sum_{b} x_{\text{III},b}\right)
\Bigg\}
\]

Subject to:

\[
\begin{align*}
& x_{\text{I},\text{Red}} < 0.10\, y_{\text{Red}} \\
& x_{\text{II},\text{Red}} > 0.50\, y_{\text{Red}} \\
& x_{\text{III},\text{Yellow}} < 0.70\, y_{\text{Yellow}} \\
& x_{\text{I},\text{Yellow}} > 0.20\, y_{\text{Yellow}} \\
& x_{\text{I},\text{Blue}} < 0.50\, y_{\text{Blue}} \\
& x_{\text{II},\text{Blue}} > 0.10\, y_{\text{Blue}} \\
& x_{\text{I},\text{Red}} + x_{\text{I},\text{Yellow}} + x_{\text{I},\text{Blue}} \leq 1500 \\
& x_{\text{II},\text{Red}} + x_{\text{II},\text{Yellow}} + x_{\text{II},\text{Blue}} \leq 2000 \\
& x_{\text{III},\text{Red}} + x_{\text{III},\text{Yellow}} + x_{\text{III},\text{Blue}} \leq 1000 \\
& y_{\text{Red}} \geq 2000 \\
& y_b = x_{\text{I},b} + x_{\text{II},b} + x_{\text{III},b} \qquad \forall b \\
& x_{g,b} \geq 0 \qquad \forall g, b
\end{align*}
\]

All identifiers, coefficients, and constraints are as retrieved and required.