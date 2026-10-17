##### Sets and Indices

- $G = \{\text{I}, \text{II}, \text{III}\}$: set of raw grades, indexed by $g$
- $B = \{\text{Red}, \text{Yellow}, \text{Blue}\}$: set of wine brands, indexed by $b$

##### Parameters

- Supply limits (kg):
  - $S_{\text{I}} = 1500$
  - $S_{\text{II}} = 2000$
  - $S_{\text{III}} = 1000$
- Unit costs (CNY/kg):
  - $c_{\text{I}} = 6$
  - $c_{\text{II}} = 4.5$
  - $c_{\text{III}} = 3$
- Selling prices (CNY/kg):
  - $p_{\text{Red}} = 5.5$
  - $p_{\text{Yellow}} = 5$
  - $p_{\text{Blue}} = 4.8$
- Blending requirements (all proportions refer to the fraction of a brand's total production from a given grade):
  - Red: Grade I $< 10\%$, Grade II $> 50\%$
  - Yellow: Grade III $< 70\%$, Grade I $> 20\%$
  - Blue: Grade I $< 50\%$, Grade II $> 10\%$
- Minimum production for Red: $Q_{\text{Red}} \geq 2000$ kg

##### Decision Variables

- $x_{g,b} \geq 0$: Amount (kg) of raw grade $g$ used in brand $b$
- $Q_b = \sum_{g \in G} x_{g,b}$: Total production (kg) of brand $b$

##### Objective Function

\[
\max \left[ \sum_{b \in B} p_b Q_b - \sum_{g \in G} c_g \left( \sum_{b \in B} x_{g,b} \right) \right]
\]

##### Constraints

1. **Blending Requirements**

   - For Red:
     - $\dfrac{x_{\text{I},\text{Red}}}{Q_{\text{Red}}} < 0.10$ (if $Q_{\text{Red}} > 0$)
     - $\dfrac{x_{\text{II},\text{Red}}}{Q_{\text{Red}}} > 0.50$ (if $Q_{\text{Red}} > 0$)
   - For Yellow:
     - $\dfrac{x_{\text{III},\text{Yellow}}}{Q_{\text{Yellow}}} < 0.70$ (if $Q_{\text{Yellow}} > 0$)
     - $\dfrac{x_{\text{I},\text{Yellow}}}{Q_{\text{Yellow}}} > 0.20$ (if $Q_{\text{Yellow}} > 0$)
   - For Blue:
     - $\dfrac{x_{\text{I},\text{Blue}}}{Q_{\text{Blue}}} < 0.50$ (if $Q_{\text{Blue}} > 0$)
     - $\dfrac{x_{\text{II},\text{Blue}}}{Q_{\text{Blue}}} > 0.10$ (if $Q_{\text{Blue}} > 0$)

   These can be written as:
   - $x_{\text{I},\text{Red}} < 0.10 Q_{\text{Red}}$
   - $x_{\text{II},\text{Red}} > 0.50 Q_{\text{Red}}$
   - $x_{\text{III},\text{Yellow}} < 0.70 Q_{\text{Yellow}}$
   - $x_{\text{I},\text{Yellow}} > 0.20 Q_{\text{Yellow}}$
   - $x_{\text{I},\text{Blue}} < 0.50 Q_{\text{Blue}}$
   - $x_{\text{II},\text{Blue}} > 0.10 Q_{\text{Blue}}$

2. **Raw Material Supply Constraints**

   For each grade $g$:
   \[
   \sum_{b \in B} x_{g,b} \leq S_g
   \]
   That is:
   - $x_{\text{I},\text{Red}} + x_{\text{I},\text{Yellow}} + x_{\text{I},\text{Blue}} \leq 1500$
   - $x_{\text{II},\text{Red}} + x_{\text{II},\text{Yellow}} + x_{\text{II},\text{Blue}} \leq 2000$
   - $x_{\text{III},\text{Red}} + x_{\text{III},\text{Yellow}} + x_{\text{III},\text{Blue}} \leq 1000$

3. **Minimum Production for Red**

   \[
   Q_{\text{Red}} = x_{\text{I},\text{Red}} + x_{\text{II},\text{Red}} + x_{\text{III},\text{Red}} \geq 2000
   \]

4. **Nonnegativity**

   \[
   x_{g,b} \geq 0 \quad \forall g \in G,\, b \in B
   \]

##### Complete Model

\[
\begin{align*}
\max\quad & 5.5\, Q_{\text{Red}} + 5\, Q_{\text{Yellow}} + 4.8\, Q_{\text{Blue}} \\
          & - \left[6(x_{\text{I},\text{Red}} + x_{\text{I},\text{Yellow}} + x_{\text{I},\text{Blue}})
          + 4.5(x_{\text{II},\text{Red}} + x_{\text{II},\text{Yellow}} + x_{\text{II},\text{Blue}})
          + 3(x_{\text{III},\text{Red}} + x_{\text{III},\text{Yellow}} + x_{\text{III},\text{Blue}})\right] \\
\text{s.t.}\quad
& x_{\text{I},\text{Red}} < 0.10\, Q_{\text{Red}} \\
& x_{\text{II},\text{Red}} > 0.50\, Q_{\text{Red}} \\
& x_{\text{III},\text{Yellow}} < 0.70\, Q_{\text{Yellow}} \\
& x_{\text{I},\text{Yellow}} > 0.20\, Q_{\text{Yellow}} \\
& x_{\text{I},\text{Blue}} < 0.50\, Q_{\text{Blue}} \\
& x_{\text{II},\text{Blue}} > 0.10\, Q_{\text{Blue}} \\
& x_{\text{I},\text{Red}} + x_{\text{I},\text{Yellow}} + x_{\text{I},\text{Blue}} \leq 1500 \\
& x_{\text{II},\text{Red}} + x_{\text{II},\text{Yellow}} + x_{\text{II},\text{Blue}} \leq 2000 \\
& x_{\text{III},\text{Red}} + x_{\text{III},\text{Yellow}} + x_{\text{III},\text{Blue}} \leq 1000 \\
& Q_{\text{Red}} \geq 2000 \\
& x_{g,b} \geq 0 \quad \forall g \in G,\, b \in B \\
& Q_b = x_{\text{I},b} + x_{\text{II},b} + x_{\text{III},b} \quad \forall b \in B
\end{align*}
\]

##### Parameters (from CSV)

- Grades: I, II, III
- Brands: Red, Yellow, Blue
- Supply limits: $S_{\text{I}}=1500$, $S_{\text{II}}=2000$, $S_{\text{III}}=1000$
- Costs: $c_{\text{I}}=6$, $c_{\text{II}}=4.5$, $c_{\text{III}}=3$
- Selling prices: $p_{\text{Red}}=5.5$, $p_{\text{Yellow}}=5$, $p_{\text{Blue}}=4.8$
- Blending requirements:
  - Red: I $<10\%$, II $>50\%$
  - Yellow: III $<70\%$, I $>20\%$
  - Blue: I $<50\%$, II $>10\%$
- Minimum Red production: $Q_{\text{Red}} \geq 2000$ kg

All variables and constraints are as above.