##### Decision Variables

Let $x_{g,b} \geq 0$ denote the amount (kg) of raw grade $g \in \{I, II, III\}$ used in wine brand $b \in \{\text{Red}, \text{Yellow}, \text{Blue}\}$.

##### Parameters

- Raw grades: $G = \{I, II, III\}$
- Brands: $B = \{\text{Red}, \text{Yellow}, \text{Blue}\}$

- Daily supply limits (kg):  
  $S_I = 1500$, $S_{II} = 2000$, $S_{III} = 1000$

- Unit costs (CNY/kg):  
  $c_I = 6$, $c_{II} = 4.5$, $c_{III} = 3$

- Selling prices (CNY/kg):  
  $p_{\text{Red}} = 5.5$, $p_{\text{Yellow}} = 5$, $p_{\text{Blue}} = 4.8$

##### Objective Function

Maximize total net profit:
\[
\max \left\{ \sum_{b \in B} p_b \left( \sum_{g \in G} x_{g,b} \right) - \sum_{g \in G} c_g \left( \sum_{b \in B} x_{g,b} \right) \right\}
\]

##### Constraints

1. **Blending Requirements**

   - **Red Brand:**
     - Proportion of I less than 10%:
       \[
       x_{I,\text{Red}} \leq 0.10 \sum_{g \in G} x_{g,\text{Red}}
       \]
     - Proportion of II more than 50%:
       \[
       x_{II,\text{Red}} \geq 0.50 \sum_{g \in G} x_{g,\text{Red}}
       \]

   - **Yellow Brand:**
     - Proportion of III less than 70%:
       \[
       x_{III,\text{Yellow}} \leq 0.70 \sum_{g \in G} x_{g,\text{Yellow}}
       \]
     - Proportion of I more than 20%:
       \[
       x_{I,\text{Yellow}} \geq 0.20 \sum_{g \in G} x_{g,\text{Yellow}}
       \]

   - **Blue Brand:**
     - Proportion of I less than 50%:
       \[
       x_{I,\text{Blue}} \leq 0.50 \sum_{g \in G} x_{g,\text{Blue}}
       \]
     - Proportion of II more than 10%:
       \[
       x_{II,\text{Blue}} \geq 0.10 \sum_{g \in G} x_{g,\text{Blue}}
       \]

2. **Raw Material Supply Constraints**
   \[
   \sum_{b \in B} x_{g,b} \leq S_g, \quad \forall g \in G
   \]
   That is:
   \begin{align*}
   x_{I,\text{Red}} + x_{I,\text{Yellow}} + x_{I,\text{Blue}} &\leq 1500 \\
   x_{II,\text{Red}} + x_{II,\text{Yellow}} + x_{II,\text{Blue}} &\leq 2000 \\
   x_{III,\text{Red}} + x_{III,\text{Yellow}} + x_{III,\text{Blue}} &\leq 1000
   \end{align*}

3. **Minimum Production of Red Brand**
   \[
   \sum_{g \in G} x_{g,\text{Red}} \geq 2000
   \]

4. **Nonnegativity**
   \[
   x_{g,b} \geq 0, \quad \forall g \in G,\, b \in B
   \]

---

###### Retrieved Information

- Raw grades: $G = \{I, II, III\}$
- Brands: $B = \{\text{Red}, \text{Yellow}, \text{Blue}\}$
- Daily supply limits: $S_I = 1500$, $S_{II} = 2000$, $S_{III} = 1000$
- Unit costs: $c_I = 6$, $c_{II} = 4.5$, $c_{III} = 3$
- Selling prices: $p_{\text{Red}} = 5.5$, $p_{\text{Yellow}} = 5$, $p_{\text{Blue}} = 4.8$
- Blending requirements:
  - Red: I < 10%, II > 50%
  - Yellow: III < 70%, I > 20%
  - Blue: I < 50%, II > 10%
- Minimum Red production: $2000$ kg

All vectors and matrices are included as above.