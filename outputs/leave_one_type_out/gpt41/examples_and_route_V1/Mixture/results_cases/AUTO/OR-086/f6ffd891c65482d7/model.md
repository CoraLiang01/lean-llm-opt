Let $x_{g,b}$ denote the amount (in kg) of raw grade $g \in \{\text{I}, \text{II}, \text{III}\}$ used in wine brand $b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$.

Let $y_b = \sum_{g} x_{g,b}$ denote the total production (kg) of brand $b$.

Parameters (from data):

- Raw grades: I, II, III
- Brands: Red, Yellow, Blue

- Daily Supply (kg): 
  - I: 1500
  - II: 2000
  - III: 1000

- Cost (CNY/kg): 
  - I: 6
  - II: 4.5
  - III: 3

- Selling Price (CNY/kg): 
  - Red: 5.5
  - Yellow: 5
  - Blue: 4.8

- Blending Requirements:
  - Red: I less than 10%, II more than 50%
  - Yellow: III less than 70%, I more than 20%
  - Blue: I less than 50%, II more than 10%

Decision variables:
- $x_{g,b} \geq 0$ for all $g \in \{\text{I}, \text{II}, \text{III}\}$, $b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$

Objective:
\[
\max \left[
  5.5 \cdot y_{\text{Red}} + 5 \cdot y_{\text{Yellow}} + 4.8 \cdot y_{\text{Blue}}
  - \left(6 \cdot \sum_b x_{\text{I},b} + 4.5 \cdot \sum_b x_{\text{II},b} + 3 \cdot \sum_b x_{\text{III},b}\right)
\right]
\]

Subject to:

Blending Requirements (for each brand):

- Red:
  - $\dfrac{x_{\text{I},\text{Red}}}{y_{\text{Red}}} < 0.10$ (if $y_{\text{Red}} > 0$)
  - $\dfrac{x_{\text{II},\text{Red}}}{y_{\text{Red}}} > 0.50$ (if $y_{\text{Red}} > 0$)

- Yellow:
  - $\dfrac{x_{\text{III},\text{Yellow}}}{y_{\text{Yellow}}} < 0.70$ (if $y_{\text{Yellow}} > 0$)
  - $\dfrac{x_{\text{I},\text{Yellow}}}{y_{\text{Yellow}}} > 0.20$ (if $y_{\text{Yellow}} > 0$)

- Blue:
  - $\dfrac{x_{\text{I},\text{Blue}}}{y_{\text{Blue}}} < 0.50$ (if $y_{\text{Blue}} > 0$)
  - $\dfrac{x_{\text{II},\text{Blue}}}{y_{\text{Blue}}} > 0.10$ (if $y_{\text{Blue}} > 0$)

Raw Material Supply Constraints:
\[
\sum_{b} x_{\text{I},b} \leq 1500
\]
\[
\sum_{b} x_{\text{II},b} \leq 2000
\]
\[
\sum_{b} x_{\text{III},b} \leq 1000
\]

Minimum Production Constraint:
\[
y_{\text{Red}} = \sum_{g} x_{g,\text{Red}} \geq 2000
\]

Nonnegativity:
\[
x_{g,b} \geq 0 \quad \forall g \in \{\text{I}, \text{II}, \text{III}\},\ b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}
\]

Summary of Blending Constraints (rewritten as linear inequalities for $y_b > 0$):

- Red:
  - $x_{\text{I},\text{Red}} < 0.10\, y_{\text{Red}}$
  - $x_{\text{II},\text{Red}} > 0.50\, y_{\text{Red}}$
- Yellow:
  - $x_{\text{III},\text{Yellow}} < 0.70\, y_{\text{Yellow}}$
  - $x_{\text{I},\text{Yellow}} > 0.20\, y_{\text{Yellow}}$
- Blue:
  - $x_{\text{I},\text{Blue}} < 0.50\, y_{\text{Blue}}$
  - $x_{\text{II},\text{Blue}} > 0.10\, y_{\text{Blue}}$

All variables are continuous and nonnegative.

Complete Model:

Maximize
\[
5.5\, y_{\text{Red}} + 5\, y_{\text{Yellow}} + 4.8\, y_{\text{Blue}}
- \left[6 \sum_b x_{\text{I},b} + 4.5 \sum_b x_{\text{II},b} + 3 \sum_b x_{\text{III},b}\right]
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
& \sum_{b} x_{\text{I},b} \leq 1500 \\
& \sum_{b} x_{\text{II},b} \leq 2000 \\
& \sum_{b} x_{\text{III},b} \leq 1000 \\
& y_{\text{Red}} \geq 2000 \\
& x_{g,b} \geq 0 \quad \forall g, b \\
& y_b = \sum_{g} x_{g,b} \quad \forall b
\end{align*}
\]