Let $x_{g,b}$ denote the amount (in kg) of raw grade $g \in \{\text{I}, \text{II}, \text{III}\}$ used in wine brand $b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$.

Let $y_b = \sum_{g} x_{g,b}$ denote the total production (kg) of brand $b$.

Parameters (from data):

- Raw grades: I, II, III
- Brands: Red, Yellow, Blue

Raw grade daily supply limits and costs:
- Grade I: supply $\leq 1500$ kg, cost = 6 CNY/kg
- Grade II: supply $\leq 2000$ kg, cost = 4.5 CNY/kg
- Grade III: supply $\leq 1000$ kg, cost = 3 CNY/kg

Brand selling prices:
- Red: 5.5 CNY/kg
- Yellow: 5 CNY/kg
- Blue: 4.8 CNY/kg

Blending requirements (from 30-2.csv):

- Red: I < 10%, II > 50%
- Yellow: III < 70%, I > 20%
- Blue: I < 50%, II > 10%

Minimum production for Red: $y_{\text{Red}} \geq 2000$ kg

---

**Objective:**
\[
\max \left[
  5.5\,y_{\text{Red}} + 5\,y_{\text{Yellow}} + 4.8\,y_{\text{Blue}}
  - \left(
    6 \sum_b x_{\text{I},b}
    + 4.5 \sum_b x_{\text{II},b}
    + 3 \sum_b x_{\text{III},b}
  \right)
\right]
\]

---

**Subject to:**

**1. Blending Requirements**

- For Red:
  - $\dfrac{x_{\text{I},\text{Red}}}{y_{\text{Red}}} < 0.10$ (if $y_{\text{Red}} > 0$)
  - $\dfrac{x_{\text{II},\text{Red}}}{y_{\text{Red}}} > 0.50$ (if $y_{\text{Red}} > 0$)

- For Yellow:
  - $\dfrac{x_{\text{III},\text{Yellow}}}{y_{\text{Yellow}}} < 0.70$ (if $y_{\text{Yellow}} > 0$)
  - $\dfrac{x_{\text{I},\text{Yellow}}}{y_{\text{Yellow}}} > 0.20$ (if $y_{\text{Yellow}} > 0$)

- For Blue:
  - $\dfrac{x_{\text{I},\text{Blue}}}{y_{\text{Blue}}} < 0.50$ (if $y_{\text{Blue}} > 0$)
  - $\dfrac{x_{\text{II},\text{Blue}}}{y_{\text{Blue}}} > 0.10$ (if $y_{\text{Blue}} > 0$)

Equivalently, for all $b$ with $y_b > 0$:
- $x_{g,b} < \alpha\, y_b$ for "less than" $\alpha$
- $x_{g,b} > \beta\, y_b$ for "more than" $\beta$

**2. Raw Material Supply Constraints**
\[
\sum_{b} x_{\text{I},b} \leq 1500
\]
\[
\sum_{b} x_{\text{II},b} \leq 2000
\]
\[
\sum_{b} x_{\text{III},b} \leq 1000
\]

**3. Minimum Production for Red**
\[
y_{\text{Red}} \geq 2000
\]

**4. Nonnegativity**
\[
x_{g,b} \geq 0 \quad \forall g \in \{\text{I}, \text{II}, \text{III}\},\ b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}
\]

**5. Definition of $y_b$**
\[
y_b = x_{\text{I},b} + x_{\text{II},b} + x_{\text{III},b} \quad \forall b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}
\]

---

**Explicit Blending Constraints:**

- Red:
  - $x_{\text{I},\text{Red}} < 0.10\, y_{\text{Red}}$
  - $x_{\text{II},\text{Red}} > 0.50\, y_{\text{Red}}$
- Yellow:
  - $x_{\text{III},\text{Yellow}} < 0.70\, y_{\text{Yellow}}$
  - $x_{\text{I},\text{Yellow}} > 0.20\, y_{\text{Yellow}}$
- Blue:
  - $x_{\text{I},\text{Blue}} < 0.50\, y_{\text{Blue}}$
  - $x_{\text{II},\text{Blue}} > 0.10\, y_{\text{Blue}}$

---

**Decision variables:** $x_{g,b} \geq 0$ (continuous, in kg)

---

**Complete Model:**

\[
\begin{align*}
\max\quad & 5.5\,y_{\text{Red}} + 5\,y_{\text{Yellow}} + 4.8\,y_{\text{Blue}}
- \left(
6 \sum_b x_{\text{I},b}
+ 4.5 \sum_b x_{\text{II},b}
+ 3 \sum_b x_{\text{III},b}
\right) \\[2ex]
\text{s.t.}\quad
& x_{\text{I},\text{Red}} < 0.10\, y_{\text{Red}} \\
& x_{\text{II},\text{Red}} > 0.50\, y_{\text{Red}} \\
& x_{\text{III},\text{Yellow}} < 0.70\, y_{\text{Yellow}} \\
& x_{\text{I},\text{Yellow}} > 0.20\, y_{\text{Yellow}} \\
& x_{\text{I},\text{Blue}} < 0.50\, y_{\text{Blue}} \\
& x_{\text{II},\text{Blue}} > 0.10\, y_{\text{Blue}} \\[2ex]
& x_{\text{I},\text{Red}} + x_{\text{I},\text{Yellow}} + x_{\text{I},\text{Blue}} \leq 1500 \\
& x_{\text{II},\text{Red}} + x_{\text{II},\text{Yellow}} + x_{\text{II},\text{Blue}} \leq 2000 \\
& x_{\text{III},\text{Red}} + x_{\text{III},\text{Yellow}} + x_{\text{III},\text{Blue}} \leq 1000 \\[2ex]
& y_{\text{Red}} \geq 2000 \\[2ex]
& y_b = x_{\text{I},b} + x_{\text{II},b} + x_{\text{III},b} \quad \forall b \in \{\text{Red}, \text{Yellow}, \text{Blue}\} \\
& x_{g,b} \geq 0 \quad \forall g, b
\end{align*}
\]