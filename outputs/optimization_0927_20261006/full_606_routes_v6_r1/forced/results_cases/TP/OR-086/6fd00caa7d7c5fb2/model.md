Let $x_{g,b}\geq 0$ be the quantity (kg) of raw grade $g\in G$ used in brand $b\in B$.

Sets and Indices:
- $G = \{\text{I}, \text{II}, \text{III}\}$ (raw grades)
- $B = \{\text{Red}, \text{Yellow}, \text{Blue}\}$ (brands)

Parameters (from data):

| Grade | Daily Supply (kg) | Cost (CNY/kg) |
|-------|-------------------|---------------|
| I     | 1500              | 6             |
| II    | 2000              | 4.5           |
| III   | 1000              | 3             |

| Brand  | Selling Price (CNY/kg) | Blending Requirements                                 |
|--------|-----------------------|-------------------------------------------------------|
| Red    | 5.5                   | I < 10%, II > 50%                                     |
| Yellow | 5                     | III < 70%, I > 20%                                    |
| Blue   | 4.8                   | I < 50%, II > 10%                                     |

Define:
- $c_g$ = cost per kg of grade $g$
- $s_g$ = daily supply limit of grade $g$
- $p_b$ = selling price per kg of brand $b$

Objective:
\[
\max \left\{ \sum_{b\in B} p_b \left(\sum_{g\in G} x_{g,b}\right) - \sum_{g\in G} c_g \left(\sum_{b\in B} x_{g,b}\right) \right\}
\]

Subject to:

1. **Blending Requirements** (for each brand):

   - For Red:
     - $\dfrac{x_{\text{I},\text{Red}}}{x_{\text{I},\text{Red}} + x_{\text{II},\text{Red}} + x_{\text{III},\text{Red}}} < 0.10$ (if denominator $>0$)
     - $\dfrac{x_{\text{II},\text{Red}}}{x_{\text{I},\text{Red}} + x_{\text{II},\text{Red}} + x_{\text{III},\text{Red}}} > 0.50$ (if denominator $>0$)

   - For Yellow:
     - $\dfrac{x_{\text{III},\text{Yellow}}}{x_{\text{I},\text{Yellow}} + x_{\text{II},\text{Yellow}} + x_{\text{III},\text{Yellow}}} < 0.70$ (if denominator $>0$)
     - $\dfrac{x_{\text{I},\text{Yellow}}}{x_{\text{I},\text{Yellow}} + x_{\text{II},\text{Yellow}} + x_{\text{III},\text{Yellow}}} > 0.20$ (if denominator $>0$)

   - For Blue:
     - $\dfrac{x_{\text{I},\text{Blue}}}{x_{\text{I},\text{Blue}} + x_{\text{II},\text{Blue}} + x_{\text{III},\text{Blue}}} < 0.50$ (if denominator $>0$)
     - $\dfrac{x_{\text{II},\text{Blue}}}{x_{\text{I},\text{Blue}} + x_{\text{II},\text{Blue}} + x_{\text{III},\text{Blue}}} > 0.10$ (if denominator $>0$)

   These can be written as:
   - For Red:
     - $x_{\text{I},\text{Red}} < 0.10 \cdot (x_{\text{I},\text{Red}} + x_{\text{II},\text{Red}} + x_{\text{III},\text{Red}})$
     - $x_{\text{II},\text{Red}} > 0.50 \cdot (x_{\text{I},\text{Red}} + x_{\text{II},\text{Red}} + x_{\text{III},\text{Red}})$
   - For Yellow:
     - $x_{\text{III},\text{Yellow}} < 0.70 \cdot (x_{\text{I},\text{Yellow}} + x_{\text{II},\text{Yellow}} + x_{\text{III},\text{Yellow}})$
     - $x_{\text{I},\text{Yellow}} > 0.20 \cdot (x_{\text{I},\text{Yellow}} + x_{\text{II},\text{Yellow}} + x_{\text{III},\text{Yellow}})$
   - For Blue:
     - $x_{\text{I},\text{Blue}} < 0.50 \cdot (x_{\text{I},\text{Blue}} + x_{\text{II},\text{Blue}} + x_{\text{III},\text{Blue}})$
     - $x_{\text{II},\text{Blue}} > 0.10 \cdot (x_{\text{I},\text{Blue}} + x_{\text{II},\text{Blue}} + x_{\text{III},\text{Blue}})$

2. **Raw Material Supply Constraints** (for each grade):
   \[
   \sum_{b\in B} x_{g,b} \leq s_g \qquad \forall g\in G
   \]
   That is:
   - $x_{\text{I},\text{Red}} + x_{\text{I},\text{Yellow}} + x_{\text{I},\text{Blue}} \leq 1500$
   - $x_{\text{II},\text{Red}} + x_{\text{II},\text{Yellow}} + x_{\text{II},\text{Blue}} \leq 2000$
   - $x_{\text{III},\text{Red}} + x_{\text{III},\text{Yellow}} + x_{\text{III},\text{Blue}} \leq 1000$

3. **Minimum Production of Red Brand**:
   \[
   x_{\text{I},\text{Red}} + x_{\text{II},\text{Red}} + x_{\text{III},\text{Red}} \geq 2000
   \]

4. **Non-negativity**:
   \[
   x_{g,b} \geq 0 \qquad \forall g\in G,\, b\in B
   \]

**Parameter values:**

- $c_{\text{I}} = 6$, $c_{\text{II}} = 4.5$, $c_{\text{III}} = 3$
- $s_{\text{I}} = 1500$, $s_{\text{II}} = 2000$, $s_{\text{III}} = 1000$
- $p_{\text{Red}} = 5.5$, $p_{\text{Yellow}} = 5$, $p_{\text{Blue}} = 4.8$

**Full Model:**

Maximize
\[
5.5(x_{\text{I},\text{Red}} + x_{\text{II},\text{Red}} + x_{\text{III},\text{Red}})
+ 5(x_{\text{I},\text{Yellow}} + x_{\text{II},\text{Yellow}} + x_{\text{III},\text{Yellow}})
+ 4.8(x_{\text{I},\text{Blue}} + x_{\text{II},\text{Blue}} + x_{\text{III},\text{Blue}})
\]
\[
- \left[
6(x_{\text{I},\text{Red}} + x_{\text{I},\text{Yellow}} + x_{\text{I},\text{Blue}})
+ 4.5(x_{\text{II},\text{Red}} + x_{\text{II},\text{Yellow}} + x_{\text{II},\text{Blue}})
+ 3(x_{\text{III},\text{Red}} + x_{\text{III},\text{Yellow}} + x_{\text{III},\text{Blue}})
\right]
\]

Subject to:

- $x_{\text{I},\text{Red}} < 0.10(x_{\text{I},\text{Red}} + x_{\text{II},\text{Red}} + x_{\text{III},\text{Red}})$
- $x_{\text{II},\text{Red}} > 0.50(x_{\text{I},\text{Red}} + x_{\text{II},\text{Red}} + x_{\text{III},\text{Red}})$
- $x_{\text{III},\text{Yellow}} < 0.70(x_{\text{I},\text{Yellow}} + x_{\text{II},\text{Yellow}} + x_{\text{III},\text{Yellow}})$
- $x_{\text{I},\text{Yellow}} > 0.20(x_{\text{I},\text{Yellow}} + x_{\text{II},\text{Yellow}} + x_{\text{III},\text{Yellow}})$
- $x_{\text{I},\text{Blue}} < 0.50(x_{\text{I},\text{Blue}} + x_{\text{II},\text{Blue}} + x_{\text{III},\text{Blue}})$
- $x_{\text{II},\text{Blue}} > 0.10(x_{\text{I},\text{Blue}} + x_{\text{II},\text{Blue}} + x_{\text{III},\text{Blue}})$

- $x_{\text{I},\text{Red}} + x_{\text{I},\text{Yellow}} + x_{\text{I},\text{Blue}} \leq 1500$
- $x_{\text{II},\text{Red}} + x_{\text{II},\text{Yellow}} + x_{\text{II},\text{Blue}} \leq 2000$
- $x_{\text{III},\text{Red}} + x_{\text{III},\text{Yellow}} + x_{\text{III},\text{Blue}} \leq 1000$

- $x_{\text{I},\text{Red}} + x_{\text{II},\text{Red}} + x_{\text{III},\text{Red}} \geq 2000$

- $x_{g,b} \geq 0$ for all $g\in G$, $b\in B$