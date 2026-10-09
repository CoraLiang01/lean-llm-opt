Let:
- $x_{i}$ = quantity of product $i$ produced, where $i \in \{\text{I}, \text{II}, \text{III}\}$, all $x_i \geq 0$ and continuous.
- $y_{A1,i}$ = amount of product $i$ processed on equipment A1 (similarly for $y_{A2,i}$, $y_{A3,i}$, $y_{B1,i}$, $y_{B2,i}$, $y_{B3,i}$, $y_{B4,i}$), all $\geq 0$ and continuous.

Parameters (from 43.csv, in source order):

Equipment / Cost | Product I | Product II | Product III | Available Equipment Operating Time | Equipment Cost at Full Load (yuan)
---|---|---|---|---|---
A1 | 5 | 10 |  | 6000 | 300
A2 | 7 | 9 | 12 | 10000 | 321
A3 | 6 | 11 | 2 | 8000 | 203
B1 | 6 | 8 |  | 4000 | 250
B2 | 4 |  | 11 | 7000 | 783
B3 | 7 |  |  | 4000 | 200
B4 | 3 | 5 | 8 | 5000 | 300
Raw Material Cost (yuan/unit) | 0.25 | 0.35 | 0.5 |  | 
Unit Price (yuan/unit) | 1.25 | 2 | 2.8 |  | 

Define:

- $c_i$ = raw material cost per unit of product $i$
- $s_i$ = selling price per unit of product $i$
- $t_{e,i}$ = processing time per unit of product $i$ on equipment $e$
- $T_e$ = available operating time for equipment $e$
- $C_e$ = equipment cost at full load for equipment $e$

Decision variables:
- $y_{e,i}$: amount of product $i$ processed on equipment $e$ (continuous, $\geq 0$)

Product-equipment assignment constraints (from description):

- Product I: A1, A2, A3 for procedure A; B1, B2, B3, B4 for procedure B
- Product II: A1, A2, A3 for procedure A; B1, B4 for procedure B
- Product III: A2, A3 for procedure A; B2, B4 for procedure B

Let $A = \{\text{A1}, \text{A2}, \text{A3}\}$, $B = \{\text{B1}, \text{B2}, \text{B3}, \text{B4}\}$

#### Objective Function

Maximize total profit:

\[
\max \left[
(s_{\text{I}} - c_{\text{I}}) x_{\text{I}} +
(s_{\text{II}} - c_{\text{II}}) x_{\text{II}} +
(s_{\text{III}} - c_{\text{III}}) x_{\text{III}}
- \sum_{e \in A \cup B} C_e \cdot \frac{1}{T_e} \cdot \left( \sum_{i} t_{e,i} y_{e,i} \right)
\right]
\]

#### Constraints

1. **Production-Processing Consistency**

For each product, the amount produced must be processed through both A and B procedures:

- For procedure A:
    - $x_{\text{I}} = \sum_{e \in A} y_{e,\text{I}}$
    - $x_{\text{II}} = \sum_{e \in A} y_{e,\text{II}}$
    - $x_{\text{III}} = \sum_{e \in \{\text{A2}, \text{A3}\}} y_{e,\text{III}}$

- For procedure B:
    - $x_{\text{I}} = \sum_{e \in B} y_{e,\text{I}}$
    - $x_{\text{II}} = y_{\text{B1},\text{II}} + y_{\text{B4},\text{II}}$
    - $x_{\text{III}} = y_{\text{B2},\text{III}} + y_{\text{B4},\text{III}}$

2. **Equipment Time Capacity**

For each equipment $e$:

\[
\sum_{i} t_{e,i} y_{e,i} \leq T_e
\]

where $t_{e,i}$ is blank if product $i$ cannot be processed on equipment $e$ (i.e., $y_{e,i} = 0$).

Explicitly, for each equipment (from source order):

- A1: $5y_{\text{A1},\text{I}} + 10y_{\text{A1},\text{II}} \leq 6000$
- A2: $7y_{\text{A2},\text{I}} + 9y_{\text{A2},\text{II}} + 12y_{\text{A2},\text{III}} \leq 10000$
- A3: $6y_{\text{A3},\text{I}} + 11y_{\text{A3},\text{II}} + 2y_{\text{A3},\text{III}} \leq 8000$
- B1: $6y_{\text{B1},\text{I}} + 8y_{\text{B1},\text{II}} \leq 4000$
- B2: $4y_{\text{B2},\text{I}} + 11y_{\text{B2},\text{III}} \leq 7000$
- B3: $7y_{\text{B3},\text{I}} \leq 4000$
- B4: $3y_{\text{B4},\text{I}} + 5y_{\text{B4},\text{II}} + 8y_{\text{B4},\text{III}} \leq 5000$

3. **Product-Equipment Assignment Restrictions**

Set $y_{e,i} = 0$ if product $i$ cannot be processed on equipment $e$ (i.e., if $t_{e,i}$ is blank in the table).

#### Variable Domains

\[
x_{\text{I}}, x_{\text{II}}, x_{\text{III}} \geq 0 \\
y_{e,i} \geq 0 \quad \text{for all allowed } (e,i)
\]

#### Parameters (from data):

- $c_{\text{I}} = 0.25$, $c_{\text{II}} = 0.35$, $c_{\text{III}} = 0.5$
- $s_{\text{I}} = 1.25$, $s_{\text{II}} = 2$, $s_{\text{III}} = 2.8$
- $C_{\text{A1}} = 300$, $C_{\text{A2}} = 321$, $C_{\text{A3}} = 203$, $C_{\text{B1}} = 250$, $C_{\text{B2}} = 783$, $C_{\text{B3}} = 200$, $C_{\text{B4}} = 300$
- $T_{\text{A1}} = 6000$, $T_{\text{A2}} = 10000$, $T_{\text{A3}} = 8000$, $T_{\text{B1}} = 4000$, $T_{\text{B2}} = 7000$, $T_{\text{B3}} = 4000$, $T_{\text{B4}} = 5000$
- $t_{e,i}$ as in the table above.

#### Complete Model

Maximize:
\[
(1.25 - 0.25)x_{\text{I}} + (2 - 0.35)x_{\text{II}} + (2.8 - 0.5)x_{\text{III}}
- \left[
\frac{300}{6000}(5y_{\text{A1},\text{I}} + 10y_{\text{A1},\text{II}})
+ \frac{321}{10000}(7y_{\text{A2},\text{I}} + 9y_{\text{A2},\text{II}} + 12y_{\text{A2},\text{III}})
+ \frac{203}{8000}(6y_{\text{A3},\text{I}} + 11y_{\text{A3},\text{II}} + 2y_{\text{A3},\text{III}})
+ \frac{250}{4000}(6y_{\text{B1},\text{I}} + 8y_{\text{B1},\text{II}})
+ \frac{783}{7000}(4y_{\text{B2},\text{I}} + 11y_{\text{B2},\text{III}})
+ \frac{200}{4000}(7y_{\text{B3},\text{I}})
+ \frac{300}{5000}(3y_{\text{B4},\text{I}} + 5y_{\text{B4},\text{II}} + 8y_{\text{B4},\text{III}})
\right]
\]

Subject to:

\[
\begin{align*}
& x_{\text{I}} = y_{\text{A1},\text{I}} + y_{\text{A2},\text{I}} + y_{\text{A3},\text{I}} \\
& x_{\text{II}} = y_{\text{A1},\text{II}} + y_{\text{A2},\text{II}} + y_{\text{A3},\text{II}} \\
& x_{\text{III}} = y_{\text{A2},\text{III}} + y_{\text{A3},\text{III}} \\
& x_{\text{I}} = y_{\text{B1},\text{I}} + y_{\text{B2},\text{I}} + y_{\text{B3},\text{I}} + y_{\text{B4},\text{I}} \\
& x_{\text{II}} = y_{\text{B1},\text{II}} + y_{\text{B4},\text{II}} \\
& x_{\text{III}} = y_{\text{B2},\text{III}} + y_{\text{B4},\text{III}} \\
& 5y_{\text{A1},\text{I}} + 10y_{\text{A1},\text{II}} \leq 6000 \\
& 7y_{\text{A2},\text{I}} + 9y_{\text{A2},\text{II}} + 12y_{\text{A2},\text{III}} \leq 10000 \\
& 6y_{\text{A3},\text{I}} + 11y_{\text{A3},\text{II}} + 2y_{\text{A3},\text{III}} \leq 8000 \\
& 6y_{\text{B1},\text{I}} + 8y_{\text{B1},\text{II}} \leq 4000 \\
& 4y_{\text{B2},\text{I}} + 11y_{\text{B2},\text{III}} \leq 7000 \\
& 7y_{\text{B3},\text{I}} \leq 4000 \\
& 3y_{\text{B4},\text{I}} + 5y_{\text{B4},\text{II}} + 8y_{\text{B4},\text{III}} \leq 5000 \\
& y_{e,i} = 0 \text{ if } t_{e,i} \text{ is blank in the table above} \\
& x_{\text{I}}, x_{\text{II}}, x_{\text{III}} \geq 0 \\
& y_{e,i} \geq 0 \text{ for all allowed } (e,i)
\end{align*}
\]