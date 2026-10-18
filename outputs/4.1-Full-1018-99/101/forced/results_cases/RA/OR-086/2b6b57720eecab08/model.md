Let $x_{g,b}$ denote the quantity (in kg) of raw grade $g \in \{\text{I}, \text{II}, \text{III}\}$ allocated to wine brand $b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$.

Parameters (from the data):

- Raw grades: I, II, III
- Brands: Red, Yellow, Blue

Raw grade daily supply limits and costs:
\[
\begin{align*}
&\text{Grade I:} \quad \text{Supply} = 1500 \text{ kg}, \quad \text{Cost} = 6 \text{ CNY/kg} \\
&\text{Grade II:} \quad \text{Supply} = 2000 \text{ kg}, \quad \text{Cost} = 4.5 \text{ CNY/kg} \\
&\text{Grade III:} \quad \text{Supply} = 1000 \text{ kg}, \quad \text{Cost} = 3 \text{ CNY/kg}
\end{align*}
\]

Brand selling prices:
\[
\begin{align*}
&\text{Red:} \quad 5.5 \text{ CNY/kg} \\
&\text{Yellow:} \quad 5 \text{ CNY/kg} \\
&\text{Blue:} \quad 4.8 \text{ CNY/kg}
\end{align*}
\]

Blending requirements (from 30-2.csv):

- Red: I less than 10%, II more than 50%
- Yellow: III less than 70%, I more than 20%
- Blue: I less than 50%, II more than 10%

Define total production of each brand:
\[
y_b = \sum_{g} x_{g,b} \qquad \forall b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}
\]

Objective:
\[
\max \left[
5.5 \cdot y_{\text{Red}} + 5 \cdot y_{\text{Yellow}} + 4.8 \cdot y_{\text{Blue}}
- \left(6 \cdot \sum_b x_{\text{I},b} + 4.5 \cdot \sum_b x_{\text{II},b} + 3 \cdot \sum_b x_{\text{III},b}\right)
\right]
\]

Subject to:

Raw material supply constraints:
\[
\begin{align*}
&\sum_{b} x_{\text{I},b} \leq 1500 \\
&\sum_{b} x_{\text{II},b} \leq 2000 \\
&\sum_{b} x_{\text{III},b} \leq 1000
\end{align*}
\]

Blending requirements (for each brand):

Red:
\[
\begin{align*}
&\frac{x_{\text{I},\text{Red}}}{y_{\text{Red}}} < 0.10 \quad \text{(I less than 10%)} \\
&\frac{x_{\text{II},\text{Red}}}{y_{\text{Red}}} > 0.50 \quad \text{(II more than 50%)}
\end{align*}
\]

Yellow:
\[
\begin{align*}
&\frac{x_{\text{III},\text{Yellow}}}{y_{\text{Yellow}}} < 0.70 \quad \text{(III less than 70%)} \\
&\frac{x_{\text{I},\text{Yellow}}}{y_{\text{Yellow}}} > 0.20 \quad \text{(I more than 20%)}
\end{align*}
\]

Blue:
\[
\begin{align*}
&\frac{x_{\text{I},\text{Blue}}}{y_{\text{Blue}}} < 0.50 \quad \text{(I less than 50%)} \\
&\frac{x_{\text{II},\text{Blue}}}{y_{\text{Blue}}} > 0.10 \quad \text{(II more than 10%)}
\end{align*}
\]

Minimum production constraint:
\[
y_{\text{Red}} \geq 2000
\]

Nonnegativity:
\[
x_{g,b} \geq 0 \qquad \forall g \in \{\text{I}, \text{II}, \text{III}\},\ b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}
\]

Summary of all constraints (rewritten as linear inequalities for implementation):

Red:
\[
\begin{align*}
&x_{\text{I},\text{Red}} - 0.10\, y_{\text{Red}} < 0 \\
&x_{\text{II},\text{Red}} - 0.50\, y_{\text{Red}} > 0
\end{align*}
\]

Yellow:
\[
\begin{align*}
&x_{\text{III},\text{Yellow}} - 0.70\, y_{\text{Yellow}} < 0 \\
&x_{\text{I},\text{Yellow}} - 0.20\, y_{\text{Yellow}} > 0
\end{align*}
\]

Blue:
\[
\begin{align*}
&x_{\text{I},\text{Blue}} - 0.50\, y_{\text{Blue}} < 0 \\
&x_{\text{II},\text{Blue}} - 0.10\, y_{\text{Blue}} > 0
\end{align*}
\]

All variables $x_{g,b} \geq 0$.

Complete Model:

\[
\begin{align*}
\max\quad & 5.5\, y_{\text{Red}} + 5\, y_{\text{Yellow}} + 4.8\, y_{\text{Blue}}
- \left[6 \sum_b x_{\text{I},b} + 4.5 \sum_b x_{\text{II},b} + 3 \sum_b x_{\text{III},b}\right] \\
\text{s.t.}\quad
& \sum_{b} x_{\text{I},b} \leq 1500 \\
& \sum_{b} x_{\text{II},b} \leq 2000 \\
& \sum_{b} x_{\text{III},b} \leq 1000 \\
& x_{\text{I},\text{Red}} - 0.10\, y_{\text{Red}} < 0 \\
& x_{\text{II},\text{Red}} - 0.50\, y_{\text{Red}} > 0 \\
& x_{\text{III},\text{Yellow}} - 0.70\, y_{\text{Yellow}} < 0 \\
& x_{\text{I},\text{Yellow}} - 0.20\, y_{\text{Yellow}} > 0 \\
& x_{\text{I},\text{Blue}} - 0.50\, y_{\text{Blue}} < 0 \\
& x_{\text{II},\text{Blue}} - 0.10\, y_{\text{Blue}} > 0 \\
& y_{\text{Red}} \geq 2000 \\
& x_{g,b} \geq 0 \qquad \forall g, b \\
& y_b = \sum_{g} x_{g,b} \qquad \forall b
\end{align*}
\]