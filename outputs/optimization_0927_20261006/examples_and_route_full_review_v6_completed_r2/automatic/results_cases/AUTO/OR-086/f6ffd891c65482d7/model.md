Let $x_{g,b}$ denote the kilograms of raw grade $g$ used in wine brand $b$, where $g \in \{\text{I}, \text{II}, \text{III}\}$ and $b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$.

Let $y_b = \sum_{g} x_{g,b}$ denote the total kilograms produced of brand $b$.

Parameters (from the data):

- Raw grades: I, II, III
- Daily supply limits: $S_{\text{I}} = 1500$, $S_{\text{II}} = 2000$, $S_{\text{III}} = 1000$
- Unit costs: $c_{\text{I}} = 6$, $c_{\text{II}} = 4.5$, $c_{\text{III}} = 3$
- Brands: Red, Yellow, Blue
- Selling prices: $p_{\text{Red}} = 5.5$, $p_{\text{Yellow}} = 5$, $p_{\text{Blue}} = 4.8$

Blending requirements (from 30-2.csv):

- Red: I < 10%, II > 50%
- Yellow: III < 70%, I > 20%
- Blue: I < 50%, II > 10%

Objective:
\[
\max \left[
  5.5 \cdot y_{\text{Red}} + 5 \cdot y_{\text{Yellow}} + 4.8 \cdot y_{\text{Blue}}
  - \left(6 \cdot \sum_b x_{\text{I},b} + 4.5 \cdot \sum_b x_{\text{II},b} + 3 \cdot \sum_b x_{\text{III},b}\right)
\right]
\]

Subject to:

1. **Raw Material Supply Constraints:**
   \[
   \sum_{b} x_{\text{I},b} \leq 1500
   \]
   \[
   \sum_{b} x_{\text{II},b} \leq 2000
   \]
   \[
   \sum_{b} x_{\text{III},b} \leq 1000
   \]

2. **Blending Requirements:**

   - For Red:
     \[
     \frac{x_{\text{I},\text{Red}}}{y_{\text{Red}}} < 0.10 \quad \text{if } y_{\text{Red}} > 0
     \]
     \[
     \frac{x_{\text{II},\text{Red}}}{y_{\text{Red}}} > 0.50 \quad \text{if } y_{\text{Red}} > 0
     \]
     (If $y_{\text{Red}} = 0$, these are vacuously satisfied.)

   - For Yellow:
     \[
     \frac{x_{\text{III},\text{Yellow}}}{y_{\text{Yellow}}} < 0.70 \quad \text{if } y_{\text{Yellow}} > 0
     \]
     \[
     \frac{x_{\text{I},\text{Yellow}}}{y_{\text{Yellow}}} > 0.20 \quad \text{if } y_{\text{Yellow}} > 0
     \]

   - For Blue:
     \[
     \frac{x_{\text{I},\text{Blue}}}{y_{\text{Blue}}} < 0.50 \quad \text{if } y_{\text{Blue}} > 0
     \]
     \[
     \frac{x_{\text{II},\text{Blue}}}{y_{\text{Blue}}} > 0.10 \quad \text{if } y_{\text{Blue}} > 0
     \]

   These can be written as:
   - For Red:
     \[
     x_{\text{I},\text{Red}} < 0.10\, y_{\text{Red}}
     \]
     \[
     x_{\text{II},\text{Red}} > 0.50\, y_{\text{Red}}
     \]
   - For Yellow:
     \[
     x_{\text{III},\text{Yellow}} < 0.70\, y_{\text{Yellow}}
     \]
     \[
     x_{\text{I},\text{Yellow}} > 0.20\, y_{\text{Yellow}}
     \]
   - For Blue:
     \[
     x_{\text{I},\text{Blue}} < 0.50\, y_{\text{Blue}}
     \]
     \[
     x_{\text{II},\text{Blue}} > 0.10\, y_{\text{Blue}}
     \]

   For each $b$, $y_b = \sum_{g} x_{g,b}$.

3. **Minimum Production Constraint:**
   \[
   y_{\text{Red}} \geq 2000
   \]

4. **Nonnegativity:**
   \[
   x_{g,b} \geq 0 \quad \forall g \in \{\text{I}, \text{II}, \text{III}\},\ b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}
   \]

All variables are continuous and nonnegative.

---

**Complete Model:**

\[
\begin{align*}
\max\ & 5.5\, y_{\text{Red}} + 5\, y_{\text{Yellow}} + 4.8\, y_{\text{Blue}}
- \left[6 \sum_b x_{\text{I},b} + 4.5 \sum_b x_{\text{II},b} + 3 \sum_b x_{\text{III},b}\right] \\
\text{s.t.}\quad
& \sum_{b} x_{\text{I},b} \leq 1500 \\
& \sum_{b} x_{\text{II},b} \leq 2000 \\
& \sum_{b} x_{\text{III},b} \leq 1000 \\
& x_{\text{I},\text{Red}} < 0.10\, y_{\text{Red}} \\
& x_{\text{II},\text{Red}} > 0.50\, y_{\text{Red}} \\
& x_{\text{III},\text{Yellow}} < 0.70\, y_{\text{Yellow}} \\
& x_{\text{I},\text{Yellow}} > 0.20\, y_{\text{Yellow}} \\
& x_{\text{I},\text{Blue}} < 0.50\, y_{\text{Blue}} \\
& x_{\text{II},\text{Blue}} > 0.10\, y_{\text{Blue}} \\
& y_{\text{Red}} \geq 2000 \\
& y_b = x_{\text{I},b} + x_{\text{II},b} + x_{\text{III},b} \quad \forall b \\
& x_{g,b} \geq 0 \quad \forall g, b
\end{align*}
\]

Where:
- $g \in \{\text{I}, \text{II}, \text{III}\}$
- $b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$

All coefficients and constraints are taken directly from the provided data.