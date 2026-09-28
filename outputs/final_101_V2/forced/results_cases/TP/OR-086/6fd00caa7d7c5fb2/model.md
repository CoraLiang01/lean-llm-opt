##### Sets and Indices

- Let $G = \{\text{I}, \text{II}, \text{III}\}$ be the set of raw grades, indexed by $g$.
- Let $B = \{\text{Red}, \text{Yellow}, \text{Blue}\}$ be the set of wine brands, indexed by $b$.

##### Parameters

From 30-1.csv (source order):

| Grade | Daily Supply (kg) | Cost (CNY/kg) |
|-------|-------------------|---------------|
| I     | 1500              | 6             |
| II    | 2000              | 4.5           |
| III   | 1000              | 3             |

From 30-2.csv (source order):

| Brand  | Blending Requirements                  | Selling Price (CNY/kg) |
|--------|----------------------------------------|------------------------|
| Red    | I less than 10%  II more than 50%      | 5.5                    |
| Yellow | III less than 70%  I more than 20%     | 5                      |
| Blue   | I less than 50%  II more than 10%      | 4.8                    |

Let:
- $S_g$ = daily supply limit of grade $g$ (kg)
- $c_g$ = unit cost of grade $g$ (CNY/kg)
- $p_b$ = selling price of brand $b$ (CNY/kg)

##### Decision Variables

- $x_{g,b} \geq 0$: amount (kg) of grade $g$ used in brand $b$.

Let $y_b = \sum_{g \in G} x_{g,b}$: total production (kg) of brand $b$.

##### Objective Function

Maximize total net profit:
\[
\max \left\{ \sum_{b \in B} p_b \left( \sum_{g \in G} x_{g,b} \right) - \sum_{g \in G} c_g \left( \sum_{b \in B} x_{g,b} \right) \right\}
\]

##### Constraints

1. **Blending Requirements** (from 30-2.csv):

   - For Red:
     - $\frac{x_{\text{I},\text{Red}}}{y_{\text{Red}}} < 0.10$ (if $y_{\text{Red}} > 0$)
     - $\frac{x_{\text{II},\text{Red}}}{y_{\text{Red}}} > 0.50$ (if $y_{\text{Red}} > 0$)
   - For Yellow:
     - $\frac{x_{\text{III},\text{Yellow}}}{y_{\text{Yellow}}} < 0.70$ (if $y_{\text{Yellow}} > 0$)
     - $\frac{x_{\text{I},\text{Yellow}}}{y_{\text{Yellow}}} > 0.20$ (if $y_{\text{Yellow}} > 0$)
   - For Blue:
     - $\frac{x_{\text{I},\text{Blue}}}{y_{\text{Blue}}} < 0.50$ (if $y_{\text{Blue}} > 0$)
     - $\frac{x_{\text{II},\text{Blue}}}{y_{\text{Blue}}} > 0.10$ (if $y_{\text{Blue}} > 0$)

   These can be written as:
   - $x_{\text{I},\text{Red}} < 0.10\, y_{\text{Red}}$
   - $x_{\text{II},\text{Red}} > 0.50\, y_{\text{Red}}$
   - $x_{\text{III},\text{Yellow}} < 0.70\, y_{\text{Yellow}}$
   - $x_{\text{I},\text{Yellow}} > 0.20\, y_{\text{Yellow}}$
   - $x_{\text{I},\text{Blue}} < 0.50\, y_{\text{Blue}}$
   - $x_{\text{II},\text{Blue}} > 0.10\, y_{\text{Blue}}$

2. **Raw Material Supply Constraints** (from 30-1.csv):

   For each $g \in G$:
   \[
   \sum_{b \in B} x_{g,b} \leq S_g
   \]
   - $x_{\text{I},\text{Red}} + x_{\text{I},\text{Yellow}} + x_{\text{I},\text{Blue}} \leq 1500$
   - $x_{\text{II},\text{Red}} + x_{\text{II},\text{Yellow}} + x_{\text{II},\text{Blue}} \leq 2000$
   - $x_{\text{III},\text{Red}} + x_{\text{III},\text{Yellow}} + x_{\text{III},\text{Blue}} \leq 1000$

3. **Minimum Production Constraint**:

   - $y_{\text{Red}} = x_{\text{I},\text{Red}} + x_{\text{II},\text{Red}} + x_{\text{III},\text{Red}} \geq 2000$

4. **Non-negativity**:

   - $x_{g,b} \geq 0$ for all $g \in G$, $b \in B$

##### Complete Model (Numerical Formulation)

Let $x_{g,b} \geq 0$ for $g \in \{\text{I},\text{II},\text{III}\}$, $b \in \{\text{Red},\text{Yellow},\text{Blue}\}$.

\[
\max \Bigg\{
5.5\, (x_{\text{I},\text{Red}} + x_{\text{II},\text{Red}} + x_{\text{III},\text{Red}})
+ 5\, (x_{\text{I},\text{Yellow}} + x_{\text{II},\text{Yellow}} + x_{\text{III},\text{Yellow}})
+ 4.8\, (x_{\text{I},\text{Blue}} + x_{\text{II},\text{Blue}} + x_{\text{III},\text{Blue}})
\]
\[
- 6\, (x_{\text{I},\text{Red}} + x_{\text{I},\text{Yellow}} + x_{\text{I},\text{Blue}})
- 4.5\, (x_{\text{II},\text{Red}} + x_{\text{II},\text{Yellow}} + x_{\text{II},\text{Blue}})
- 3\, (x_{\text{III},\text{Red}} + x_{\text{III},\text{Yellow}} + x_{\text{III},\text{Blue}})
\Bigg\}
\]

Subject to:

Blending requirements:
- $x_{\text{I},\text{Red}} < 0.10\, (x_{\text{I},\text{Red}} + x_{\text{II},\text{Red}} + x_{\text{III},\text{Red}})$
- $x_{\text{II},\text{Red}} > 0.50\, (x_{\text{I},\text{Red}} + x_{\text{II},\text{Red}} + x_{\text{III},\text{Red}})$
- $x_{\text{III},\text{Yellow}} < 0.70\, (x_{\text{I},\text{Yellow}} + x_{\text{II},\text{Yellow}} + x_{\text{III},\text{Yellow}})$
- $x_{\text{I},\text{Yellow}} > 0.20\, (x_{\text{I},\text{Yellow}} + x_{\text{II},\text{Yellow}} + x_{\text{III},\text{Yellow}})$
- $x_{\text{I},\text{Blue}} < 0.50\, (x_{\text{I},\text{Blue}} + x_{\text{II},\text{Blue}} + x_{\text{III},\text{Blue}})$
- $x_{\text{II},\text{Blue}} > 0.10\, (x_{\text{I},\text{Blue}} + x_{\text{II},\text{Blue}} + x_{\text{III},\text{Blue}})$

Raw material supply:
- $x_{\text{I},\text{Red}} + x_{\text{I},\text{Yellow}} + x_{\text{I},\text{Blue}} \leq 1500$
- $x_{\text{II},\text{Red}} + x_{\text{II},\text{Yellow}} + x_{\text{II},\text{Blue}} \leq 2000$
- $x_{\text{III},\text{Red}} + x_{\text{III},\text{Yellow}} + x_{\text{III},\text{Blue}} \leq 1000$

Minimum production:
- $x_{\text{I},\text{Red}} + x_{\text{II},\text{Red}} + x_{\text{III},\text{Red}} \geq 2000$

Non-negativity:
- $x_{g,b} \geq 0$ for all $g, b$

All coefficients and identifiers are preserved as in the source data.