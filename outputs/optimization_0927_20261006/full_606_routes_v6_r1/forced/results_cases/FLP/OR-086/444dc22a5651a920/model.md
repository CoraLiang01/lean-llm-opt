##### Sets and Indices

- Let $G = \{\text{I}, \text{II}, \text{III}\}$ be the set of raw grades, indexed by $g$.
- Let $B = \{\text{Red}, \text{Yellow}, \text{Blue}\}$ be the set of wine brands, indexed by $b$.

##### Parameters

- Daily supply limits (kg): 
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

- Blending requirements (all proportions are relative to total production of each brand):

  - Red: 
    - Grade I: $x_{\text{I,Red}} / y_{\text{Red}} < 0.10$
    - Grade II: $x_{\text{II,Red}} / y_{\text{Red}} > 0.50$
  - Yellow:
    - Grade III: $x_{\text{III,Yellow}} / y_{\text{Yellow}} < 0.70$
    - Grade I: $x_{\text{I,Yellow}} / y_{\text{Yellow}} > 0.20$
  - Blue:
    - Grade I: $x_{\text{I,Blue}} / y_{\text{Blue}} < 0.50$
    - Grade II: $x_{\text{II,Blue}} / y_{\text{Blue}} > 0.10$

##### Decision Variables

- $x_{g,b} \geq 0$: Amount (kg) of grade $g$ used in brand $b$.
- $y_b = \sum_{g \in G} x_{g,b}$: Total production (kg) of brand $b$.

##### Objective Function

\[
\max \left[ \sum_{b \in B} p_b y_b - \sum_{g \in G} c_g \left( \sum_{b \in B} x_{g,b} \right) \right]
\]

##### Constraints

1. **Blending Requirements**

   - Red:
     - $\dfrac{x_{\text{I,Red}}}{y_{\text{Red}}} < 0.10 \implies x_{\text{I,Red}} \leq 0.10\, y_{\text{Red}}$
     - $\dfrac{x_{\text{II,Red}}}{y_{\text{Red}}} > 0.50 \implies x_{\text{II,Red}} \geq 0.50\, y_{\text{Red}}$
   - Yellow:
     - $\dfrac{x_{\text{III,Yellow}}}{y_{\text{Yellow}}} < 0.70 \implies x_{\text{III,Yellow}} \leq 0.70\, y_{\text{Yellow}}$
     - $\dfrac{x_{\text{I,Yellow}}}{y_{\text{Yellow}}} > 0.20 \implies x_{\text{I,Yellow}} \geq 0.20\, y_{\text{Yellow}}$
   - Blue:
     - $\dfrac{x_{\text{I,Blue}}}{y_{\text{Blue}}} < 0.50 \implies x_{\text{I,Blue}} \leq 0.50\, y_{\text{Blue}}$
     - $\dfrac{x_{\text{II,Blue}}}{y_{\text{Blue}}} > 0.10 \implies x_{\text{II,Blue}} \geq 0.10\, y_{\text{Blue}}$

2. **Raw Material Supply Limits**

   - For each $g \in G$:
     \[
     \sum_{b \in B} x_{g,b} \leq S_g
     \]
     That is,
     - $x_{\text{I,Red}} + x_{\text{I,Yellow}} + x_{\text{I,Blue}} \leq 1500$
     - $x_{\text{II,Red}} + x_{\text{II,Yellow}} + x_{\text{II,Blue}} \leq 2000$
     - $x_{\text{III,Red}} + x_{\text{III,Yellow}} + x_{\text{III,Blue}} \leq 1000$

3. **Minimum Production for Red Brand**

   \[
   y_{\text{Red}} \geq 2000
   \]
   where $y_{\text{Red}} = x_{\text{I,Red}} + x_{\text{II,Red}} + x_{\text{III,Red}}$

4. **Nonnegativity**

   \[
   x_{g,b} \geq 0 \quad \forall g \in G,\, b \in B
   \]

##### Complete Model

\[
\begin{align*}
\max\quad & 5.5\, y_{\text{Red}} + 5\, y_{\text{Yellow}} + 4.8\, y_{\text{Blue}} \\
          & - \left[ 6(x_{\text{I,Red}} + x_{\text{I,Yellow}} + x_{\text{I,Blue}}) + 4.5(x_{\text{II,Red}} + x_{\text{II,Yellow}} + x_{\text{II,Blue}}) + 3(x_{\text{III,Red}} + x_{\text{III,Yellow}} + x_{\text{III,Blue}}) \right] \\
\text{s.t.}\quad
& x_{\text{I,Red}} \leq 0.10\, y_{\text{Red}} \\
& x_{\text{II,Red}} \geq 0.50\, y_{\text{Red}} \\
& x_{\text{III,Yellow}} \leq 0.70\, y_{\text{Yellow}} \\
& x_{\text{I,Yellow}} \geq 0.20\, y_{\text{Yellow}} \\
& x_{\text{I,Blue}} \leq 0.50\, y_{\text{Blue}} \\
& x_{\text{II,Blue}} \geq 0.10\, y_{\text{Blue}} \\
& x_{\text{I,Red}} + x_{\text{I,Yellow}} + x_{\text{I,Blue}} \leq 1500 \\
& x_{\text{II,Red}} + x_{\text{II,Yellow}} + x_{\text{II,Blue}} \leq 2000 \\
& x_{\text{III,Red}} + x_{\text{III,Yellow}} + x_{\text{III,Blue}} \leq 1000 \\
& y_{\text{Red}} = x_{\text{I,Red}} + x_{\text{II,Red}} + x_{\text{III,Red}} \geq 2000 \\
& y_{\text{Yellow}} = x_{\text{I,Yellow}} + x_{\text{II,Yellow}} + x_{\text{III,Yellow}} \\
& y_{\text{Blue}} = x_{\text{I,Blue}} + x_{\text{II,Blue}} + x_{\text{III,Blue}} \\
& x_{g,b} \geq 0 \quad \forall g \in G,\, b \in B
\end{align*}
\]

##### Retrieved Information

- Raw grades: I, II, III
- Daily supply limits (kg): I: 1500, II: 2000, III: 1000
- Unit costs (CNY/kg): I: 6, II: 4.5, III: 3
- Brands: Red, Yellow, Blue
- Selling prices (CNY/kg): Red: 5.5, Yellow: 5, Blue: 4.8
- Blending requirements:
  - Red: I < 10%, II > 50%
  - Yellow: III < 70%, I > 20%
  - Blue: I < 50%, II > 10%
- Minimum production for Red: 2000 kg

All variables and constraints are as specified in the user query and the provided data.