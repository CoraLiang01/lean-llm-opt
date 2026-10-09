Let $x_{g,b}$ denote the kilograms of raw grade $g \in \{\text{I}, \text{II}, \text{III}\}$ used in wine brand $b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$.

Define:
- $c_g$: unit cost of grade $g$ (CNY/kg)
- $s_g$: daily supply limit of grade $g$ (kg)
- $p_b$: selling price of brand $b$ (CNY/kg)

Parameters (from data):

- $c_{\text{I}} = 6$, $s_{\text{I}} = 1500$
- $c_{\text{II}} = 4.5$, $s_{\text{II}} = 2000$
- $c_{\text{III}} = 3$, $s_{\text{III}} = 1000$
- $p_{\text{Red}} = 5.5$
- $p_{\text{Yellow}} = 5$
- $p_{\text{Blue}} = 4.8$

Let $y_b = \sum_{g} x_{g,b}$ be the total production (kg) of brand $b$.

Objective:
\[
\max \left[ \sum_{b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}} p_b \cdot y_b - \sum_{g \in \{\text{I}, \text{II}, \text{III}\}} c_g \cdot \sum_{b} x_{g,b} \right]
\]

Subject to:

##### 1. Blending Requirements

- For Red:
  - Proportion of I: $\frac{x_{\text{I},\text{Red}}}{y_{\text{Red}}} < 0.10$ (i.e., $x_{\text{I},\text{Red}} < 0.10\, y_{\text{Red}}$)
  - Proportion of II: $\frac{x_{\text{II},\text{Red}}}{y_{\text{Red}}} > 0.50$ (i.e., $x_{\text{II},\text{Red}} > 0.50\, y_{\text{Red}}$)

- For Yellow:
  - Proportion of III: $\frac{x_{\text{III},\text{Yellow}}}{y_{\text{Yellow}}} < 0.70$ (i.e., $x_{\text{III},\text{Yellow}} < 0.70\, y_{\text{Yellow}}$)
  - Proportion of I: $\frac{x_{\text{I},\text{Yellow}}}{y_{\text{Yellow}}} > 0.20$ (i.e., $x_{\text{I},\text{Yellow}} > 0.20\, y_{\text{Yellow}}$)

- For Blue:
  - Proportion of I: $\frac{x_{\text{I},\text{Blue}}}{y_{\text{Blue}}} < 0.50$ (i.e., $x_{\text{I},\text{Blue}} < 0.50\, y_{\text{Blue}}$)
  - Proportion of II: $\frac{x_{\text{II},\text{Blue}}}{y_{\text{Blue}}} > 0.10$ (i.e., $x_{\text{II},\text{Blue}} > 0.10\, y_{\text{Blue}}$)

##### 2. Raw Material Supply Constraints

\[
\sum_{b} x_{g,b} \leq s_g \qquad \forall g \in \{\text{I}, \text{II}, \text{III}\}
\]

##### 3. Minimum Production of Red

\[
y_{\text{Red}} \geq 2000
\]

##### 4. Nonnegativity

\[
x_{g,b} \geq 0 \qquad \forall g, b
\]

##### Complete Model

\[
\begin{align*}
\max\quad & 5.5\, y_{\text{Red}} + 5\, y_{\text{Yellow}} + 4.8\, y_{\text{Blue}} \\
          & - \left[6 \sum_{b} x_{\text{I},b} + 4.5 \sum_{b} x_{\text{II},b} + 3 \sum_{b} x_{\text{III},b}\right] \\
\text{s.t.}\quad
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
& x_{g,b} \geq 0 \qquad \forall g, b \\
& y_b = x_{\text{I},b} + x_{\text{II},b} + x_{\text{III},b} \qquad \forall b
\end{align*}
\]

where $g \in \{\text{I}, \text{II}, \text{III}\}$ and $b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$.