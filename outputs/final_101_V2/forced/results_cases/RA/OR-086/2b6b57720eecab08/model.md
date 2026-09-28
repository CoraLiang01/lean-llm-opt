Let $x_{g,b}$ denote the amount (kg) of raw grade $g \in \{\text{I}, \text{II}, \text{III}\}$ used in brand $b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$.

Let $y_b = \sum_{g} x_{g,b}$ denote the total production (kg) of brand $b$.

Parameters (from 30-1.csv and 30-2.csv, source order preserved):

- Raw grades: I, II, III
- Daily Supply (kg): I: 1500, II: 2000, III: 1000
- Cost (CNY/kg): I: 6, II: 4.5, III: 3

- Brands: Red, Yellow, Blue
- Selling Price (CNY/kg): Red: 5.5, Yellow: 5, Blue: 4.8

Blending Requirements (from 30-2.csv):

- Red: I less than 10%, II more than 50%
- Yellow: III less than 70%, I more than 20%
- Blue: I less than 50%, II more than 10%

Model:

Objective:
\[
\max \left[
  5.5 \cdot y_{\text{Red}} + 5 \cdot y_{\text{Yellow}} + 4.8 \cdot y_{\text{Blue}}
  - \left(6 \cdot \sum_b x_{\text{I},b} + 4.5 \cdot \sum_b x_{\text{II},b} + 3 \cdot \sum_b x_{\text{III},b}\right)
\right]
\]

Subject to:

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

Blending Requirements (for each brand):

Red:
\[
\frac{x_{\text{I},\text{Red}}}{y_{\text{Red}}} < 0.10 \qquad \text{(I less than 10\%)}
\]
\[
\frac{x_{\text{II},\text{Red}}}{y_{\text{Red}}} > 0.50 \qquad \text{(II more than 50\%)}
\]

Yellow:
\[
\frac{x_{\text{III},\text{Yellow}}}{y_{\text{Yellow}}} < 0.70 \qquad \text{(III less than 70\%)}
\]
\[
\frac{x_{\text{I},\text{Yellow}}}{y_{\text{Yellow}}} > 0.20 \qquad \text{(I more than 20\%)}
\]

Blue:
\[
\frac{x_{\text{I},\text{Blue}}}{y_{\text{Blue}}} < 0.50 \qquad \text{(I less than 50\%)}
\]
\[
\frac{x_{\text{II},\text{Blue}}}{y_{\text{Blue}}} > 0.10 \qquad \text{(II more than 10\%)}
\]

Minimum Production Constraint:
\[
y_{\text{Red}} \geq 2000
\]

Nonnegativity:
\[
x_{g,b} \geq 0 \qquad \forall g \in \{\text{I}, \text{II}, \text{III}\},\ b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}
\]

Auxiliary definitions:
\[
y_b = x_{\text{I},b} + x_{\text{II},b} + x_{\text{III},b} \qquad \forall b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}
\]

All parameters and constraints are taken directly from the provided data, preserving source order and identifiers.