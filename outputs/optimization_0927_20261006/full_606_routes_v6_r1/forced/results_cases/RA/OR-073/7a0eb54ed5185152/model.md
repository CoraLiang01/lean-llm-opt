Let $x_{i}$ denote the production quantity of product $i$ (continuous, $x_{i} \geq 0$), where $i \in \{\text{I}, \text{II}, \text{III}\}$.  
Let $y_{e,i}$ denote the amount of product $i$ processed on equipment $e$ (continuous, $y_{e,i} \geq 0$), for all eligible $(e,i)$ pairs.

Define:
- $p_i$: unit selling price of product $i$
- $r_i$: unit raw material cost of product $i$
- $t_{e,i}$: processing time per unit of product $i$ on equipment $e$
- $T_e$: available operating time for equipment $e$
- $C_e$: equipment cost at full load for equipment $e$

#### Data (from 43.csv, in source order):

| Equipment / Cost | Product I | Product II | Product III | Available Equipment Operating Time | Equipment Cost at Full Load (yuan) |
|------------------|-----------|------------|-------------|-----------------------------------|------------------------------------|
| A1               | 5         | 10         |             | 6000                              | 300                                |
| A2               | 7         | 9          | 12          | 10000                             | 321                                |
| A3               | 6         | 11         | 2           | 8000                              | 203                                |
| B1               | 6         | 8          |             | 4000                              | 250                                |
| B2               | 4         |            | 11          | 7000                              | 783                                |
| B3               | 7         |            |             | 4000                              | 200                                |
| B4               | 3         | 5          | 8           | 5000                              | 300                                |
| Raw Material Cost (yuan/unit) | 0.25      | 0.35        | 0.5         |                                   |                                    |
| Unit Price (yuan/unit)        | 1.25      | 2           | 2.8         |                                   |                                    |

#### Eligibility (from user description and data):

- Product I: can be processed on any A or B equipment where a processing time is given.
- Product II: can be processed on any A equipment where a processing time is given, and only on B1 and B4 for B equipment.
- Product III: can only be processed on A2, A3, B2, and B4.

#### Variables:

- $x_{\text{I}}, x_{\text{II}}, x_{\text{III}} \geq 0$ (total production of each product)
- $y_{e,i} \geq 0$ for all eligible $(e,i)$ (amount of product $i$ processed on equipment $e$)

#### Objective Function:

\[
\max \left[
  (1.25 - 0.25)x_{\text{I}} +
  (2 - 0.35)x_{\text{II}} +
  (2.8 - 0.5)x_{\text{III}}
  - \sum_{e} C_e \cdot \left( \frac{\sum_{i} t_{e,i} y_{e,i}}{T_e} \right)
\right]
\]

where $C_e$ is the equipment cost at full load for equipment $e$, and the term in parentheses is the fraction of full load used.

#### Constraints:

1. **Production-Processing Balance:**  
   For each product, total production equals total processed across all eligible equipment:
   \[
   x_{\text{I}} = \sum_{e \in E_{\text{I}}} y_{e,\text{I}}
   \]
   \[
   x_{\text{II}} = \sum_{e \in E_{\text{II}}} y_{e,\text{II}}
   \]
   \[
   x_{\text{III}} = \sum_{e \in E_{\text{III}}} y_{e,\text{III}}
   \]
   where $E_{i}$ is the set of equipment eligible for product $i$.

2. **Equipment Time Capacity:**  
   For each equipment $e$:
   \[
   \sum_{i} t_{e,i} y_{e,i} \leq T_e
   \]
   where the sum is over all products $i$ eligible on $e$.

3. **Eligibility Constraints:**  
   $y_{e,i} = 0$ if product $i$ is not eligible on equipment $e$ (i.e., if $t_{e,i}$ is blank in the table).

#### Explicit Data Mapping:

- $t_{e,i}$: processing time per unit (from table, blank means not eligible)
- $T_e$: available equipment operating time (from table)
- $C_e$: equipment cost at full load (from table)
- $p_i$: unit price (last row)
- $r_i$: raw material cost (second to last row)

#### Eligible $(e,i)$ pairs and $t_{e,i}$:

- A1: I (5), II (10)
- A2: I (7), II (9), III (12)
- A3: I (6), II (11), III (2)
- B1: I (6), II (8)
- B2: I (4), III (11)
- B3: I (7)
- B4: I (3), II (5), III (8)

#### Full Model:

**Variables:**
- $x_{\text{I}}, x_{\text{II}}, x_{\text{III}} \geq 0$
- $y_{e,i} \geq 0$ for all eligible $(e,i)$

**Objective:**
\[
\max \Bigg[
  (1.25 - 0.25)x_{\text{I}} +
  (2 - 0.35)x_{\text{II}} +
  (2.8 - 0.5)x_{\text{III}}
  - \Bigg(
    300 \cdot \frac{5y_{A1,I} + 10y_{A1,II}}{6000}
    + 321 \cdot \frac{7y_{A2,I} + 9y_{A2,II} + 12y_{A2,III}}{10000}
    + 203 \cdot \frac{6y_{A3,I} + 11y_{A3,II} + 2y_{A3,III}}{8000}
    + 250 \cdot \frac{6y_{B1,I} + 8y_{B1,II}}{4000}
    + 783 \cdot \frac{4y_{B2,I} + 11y_{B2,III}}{7000}
    + 200 \cdot \frac{7y_{B3,I}}{4000}
    + 300 \cdot \frac{3y_{B4,I} + 5y_{B4,II} + 8y_{B4,III}}{5000}
  \Bigg)
\Bigg]
\]

**Subject to:**

Production-Processing Balance:
\[
x_{\text{I}} = y_{A1,I} + y_{A2,I} + y_{A3,I} + y_{B1,I} + y_{B2,I} + y_{B3,I} + y_{B4,I}
\]
\[
x_{\text{II}} = y_{A1,II} + y_{A2,II} + y_{A3,II} + y_{B1,II} + y_{B4,II}
\]
\[
x_{\text{III}} = y_{A2,III} + y_{A3,III} + y_{B2,III} + y_{B4,III}
\]

Equipment Time Capacity:
\[
5y_{A1,I} + 10y_{A1,II} \leq 6000
\]
\[
7y_{A2,I} + 9y_{A2,II} + 12y_{A2,III} \leq 10000
\]
\[
6y_{A3,I} + 11y_{A3,II} + 2y_{A3,III} \leq 8000
\]
\[
6y_{B1,I} + 8y_{B1,II} \leq 4000
\]
\[
4y_{B2,I} + 11y_{B2,III} \leq 7000
\]
\[
7y_{B3,I} \leq 4000
\]
\[
3y_{B4,I} + 5y_{B4,II} + 8y_{B4,III} \leq 5000
\]

Nonnegativity:
\[
x_{\text{I}}, x_{\text{II}}, x_{\text{III}} \geq 0
\]
\[
y_{e,i} \geq 0 \quad \text{for all eligible } (e,i)
\]

And $y_{e,i} = 0$ for all ineligible $(e,i)$ pairs (i.e., where $t_{e,i}$ is blank).

This is a complete continuous linear program maximizing profit, using all data and eligibility constraints as specified.