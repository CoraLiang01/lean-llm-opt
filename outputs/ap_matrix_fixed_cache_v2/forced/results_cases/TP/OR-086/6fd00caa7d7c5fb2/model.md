##### Sets and Indices

Let $G = \{\text{I}, \text{II}, \text{III}\}$ be the set of raw grades.

Let $B = \{\text{Red}, \text{Yellow}, \text{Blue}\}$ be the set of wine brands.

Let $x_{g,b} \geq 0$ be the amount (kg) of raw grade $g \in G$ used in brand $b \in B$.

Let $y_b = \sum_{g \in G} x_{g,b}$ be the total production (kg) of brand $b$.

##### Parameters (from retrieved data)

- Raw grade daily supply limits and costs:
  - Grade I: supply $= 1500$ kg, cost $= 6$ CNY/kg
  - Grade II: supply $= 2000$ kg, cost $= 4.5$ CNY/kg
  - Grade III: supply $= 1000$ kg, cost $= 3$ CNY/kg

- Brand selling prices and blending requirements:
  - Red: price $= 5.5$ CNY/kg; I $<$ 10%, II $>$ 50%
  - Yellow: price $= 5$ CNY/kg; III $<$ 70%, I $>$ 20%
  - Blue: price $= 4.8$ CNY/kg; I $<$ 50%, II $>$ 10%

##### Decision Variables

$x_{g,b} \geq 0$ for all $g \in G$, $b \in B$

##### Objective Function

\[
\max \left\{
\begin{aligned}
&\text{Total Net Profit} = \text{Total Revenue} - \text{Total Raw Material Cost} \\
&= \sum_{b \in B} p_b \left( \sum_{g \in G} x_{g,b} \right) - \sum_{g \in G} c_g \left( \sum_{b \in B} x_{g,b} \right)
\end{aligned}
\right.
\]

Where:
- $p_{\text{Red}} = 5.5$, $p_{\text{Yellow}} = 5$, $p_{\text{Blue}} = 4.8$
- $c_{\text{I}} = 6$, $c_{\text{II}} = 4.5$, $c_{\text{III}} = 3$

##### Constraints

1. **Blending Requirements (proportion constraints for each brand):**

   - For Red:
     - $\dfrac{x_{\text{I},\text{Red}}}{y_{\text{Red}}} < 0.10$ (if $y_{\text{Red}} > 0$)
     - $\dfrac{x_{\text{II},\text{Red}}}{y_{\text{Red}}} > 0.50$ (if $y_{\text{Red}} > 0$)

   - For Yellow:
     - $\dfrac{x_{\text{III},\text{Yellow}}}{y_{\text{Yellow}}} < 0.70$ (if $y_{\text{Yellow}} > 0$)
     - $\dfrac{x_{\text{I},\text{Yellow}}}{y_{\text{Yellow}}} > 0.20$ (if $y_{\text{Yellow}} > 0$)

   - For Blue:
     - $\dfrac{x_{\text{I},\text{Blue}}}{y_{\text{Blue}}} < 0.50$ (if $y_{\text{Blue}} > 0$)
     - $\dfrac{x_{\text{II},\text{Blue}}}{y_{\text{Blue}}} > 0.10$ (if $y_{\text{Blue}} > 0$)

   These can be written as:
   - $x_{\text{I},\text{Red}} < 0.10\, y_{\text{Red}}$
   - $x_{\text{II},\text{Red}} > 0.50\, y_{\text{Red}}$
   - $x_{\text{III},\text{Yellow}} < 0.70\, y_{\text{Yellow}}$
   - $x_{\text{I},\text{Yellow}} > 0.20\, y_{\text{Yellow}}$
   - $x_{\text{I},\text{Blue}} < 0.50\, y_{\text{Blue}}$
   - $x_{\text{II},\text{Blue}} > 0.10\, y_{\text{Blue}}$

2. **Raw Material Supply Constraints:**

   - $\sum_{b \in B} x_{\text{I},b} \leq 1500$
   - $\sum_{b \in B} x_{\text{II},b} \leq 2000$
   - $\sum_{b \in B} x_{\text{III},b} \leq 1000$

3. **Minimum Production of Red Brand:**

   - $y_{\text{Red}} = \sum_{g \in G} x_{g,\text{Red}} \geq 2000$

4. **Non-negativity:**

   - $x_{g,b} \geq 0$ for all $g \in G$, $b \in B$

##### Complete Model

\[
\begin{align*}
\max\quad & 5.5\, y_{\text{Red}} + 5\, y_{\text{Yellow}} + 4.8\, y_{\text{Blue}} \\
          & - \left[6 \sum_{b \in B} x_{\text{I},b} + 4.5 \sum_{b \in B} x_{\text{II},b} + 3 \sum_{b \in B} x_{\text{III},b}\right] \\
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
& x_{g,b} \geq 0 \quad \forall g \in G,\, b \in B \\
& y_b = x_{\text{I},b} + x_{\text{II},b} + x_{\text{III},b} \quad \forall b \in B
\end{align*}
\]

##### Retrieved Information

{
  "grades": [
    {"Grade": "I", "Daily Supply (kg)": 1500, "Cost (CNY/kg)": 6},
    {"Grade": "II", "Daily Supply (kg)": 2000, "Cost (CNY/kg)": 4.5},
    {"Grade": "III", "Daily Supply (kg)": 1000, "Cost (CNY/kg)": 3}
  ],
  "brands": [
    {"Brand": "Red", "Blending Requirements": "I less than 10%  II more than 50%", "Selling Price (CNY/kg)": 5.5},
    {"Brand": "Yellow", "Blending Requirements": "III less than 70%  I more than 20%", "Selling Price (CNY/kg)": 5},
    {"Brand": "Blue", "Blending Requirements": "I less than 50%  II more than 10%", "Selling Price (CNY/kg)": 4.8}
  ]
}