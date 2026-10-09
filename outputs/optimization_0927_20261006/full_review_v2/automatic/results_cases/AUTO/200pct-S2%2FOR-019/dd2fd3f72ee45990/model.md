##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

##### Parameters

- Customer demands:
  - $d_{\text{demand1}} = 9$
  - $d_{\text{demand2}} = 66$
  - $d_{\text{demand3}} = 56$
  - $d_{\text{demand4}} = 17$
  - $d_{\text{demand5}} = 43$
  - $d_{\text{demand6}} = 62$
  - $d_{\text{demand7}} = 10$
  - $d_{\text{demand8}} = 37$

- Supplier capacities:
  - $s_{\text{supplier1}} = 60$
  - $s_{\text{supplier2}} = 22$
  - $s_{\text{supplier3}} = 16$
  - $s_{\text{supplier4}} = 14$
  - $s_{\text{supplier5}} = 19$
  - $s_{\text{supplier6}} = 70$
  - $s_{\text{supplier7}} = 60$
  - $s_{\text{supplier8}} = 39$

- Transportation costs $c_{ij}$ (per unit from supplier $i$ to customer $j$):

|            | demand1      | demand2      | demand3      | demand4      | demand5      | demand6      | demand7      | demand8      |
|------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| supply1    | 0.0302073664 | 229.50723505 | 198.62356558 | 12.99505064  | 211.20732124 | 134.94429850 | 9.8222063988 | 11.394077543 |
| supply2    | 232.34691308 | 3.6258726439 | 0.2860543415 | 45.73127693  | 2.8304796563 | 107.05891033 | 299.96317913 | 23.799354363 |
| supply3    | 11.061938334 | 0.2041995327 | 0.2789447278 | 45.72191272  | 59.548955657 | 5.0975367396 | 300.00118415 | 23.711282708 |
| supply4    | 235.17948357 | 43.794668963 | 40.709846783 | 0.0777449662 | 4.2377281834 | 131.70915517 | 296.55587568 | 29.810940018 |
| supply5    | 211.85808746 | 47.601808765 | 50.040077162 | 86.14548807  | 0.0619789792 | 5.3345515296 | 270.06290424 | 3.8539331340 |
| supply6    | 6.4550663355 | 88.163236234 | 5.0470916716 | 151.46120287 | 5.2907601611 | 0.0460220534 | 9.9367066018 | 103.75460989 |
| supply7    | 174.27229047 | 250.58223529 | 253.90413042 | 16.235467318 | 12.643140515 | 175.06728241 | 2.9838396253 | 317.06551939 |
| supply8    | 207.87006254 | 1.5171684715 | 24.027239288 | 27.13399928  | 73.206724689 | 125.72910360 | 15.463103252 | 0.2016498751 |

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. Demand satisfaction (each customer receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$

2. Supply capacity (each supplier ships no more than its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$

3. Non-negativity:
   $$
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   $$

##### Complete Numerical Formulation

Let $x_{ij}$ be the quantity shipped from supplier $i$ to customer $j$.

Minimize:
\[
\begin{align*}
\min\ & \sum_{j=1}^8 \Big[ 
  0.0302073664\,x_{\text{supply1},j} + 232.34691308\,x_{\text{supply2},j} + 11.061938334\,x_{\text{supply3},j} + 235.17948357\,x_{\text{supply4},j} \\
  &\quad + 211.85808746\,x_{\text{supply5},j} + 6.4550663355\,x_{\text{supply6},j} + 174.27229047\,x_{\text{supply7},j} + 207.87006254\,x_{\text{supply8},j} \Big] \\
&+ \sum_{j=2}^8 \Big[ 
  229.50723505\,x_{\text{supply1},\text{demand2}} + 3.6258726439\,x_{\text{supply2},\text{demand2}} + 0.2041995327\,x_{\text{supply3},\text{demand2}} + 43.794668963\,x_{\text{supply4},\text{demand2}} \\
  &\quad + 47.601808765\,x_{\text{supply5},\text{demand2}} + 88.163236234\,x_{\text{supply6},\text{demand2}} + 250.58223529\,x_{\text{supply7},\text{demand2}} + 1.5171684715\,x_{\text{supply8},\text{demand2}} \Big] \\
&+ \ldots \text{(continue for all $i,j$ as per the table above)}
\end{align*}
\]

Subject to:

For each customer $j$:
\[
\sum_{i \in I} x_{ij} \geq d_j
\]
where
\[
\begin{align*}
d_{\text{demand1}} &= 9 \\
d_{\text{demand2}} &= 66 \\
d_{\text{demand3}} &= 56 \\
d_{\text{demand4}} &= 17 \\
d_{\text{demand5}} &= 43 \\
d_{\text{demand6}} &= 62 \\
d_{\text{demand7}} &= 10 \\
d_{\text{demand8}} &= 37 \\
\end{align*}
\]

For each supplier $i$:
\[
\sum_{j \in J} x_{ij} \leq s_i
\]
where
\[
\begin{align*}
s_{\text{supplier1}} &= 60 \\
s_{\text{supplier2}} &= 22 \\
s_{\text{supplier3}} &= 16 \\
s_{\text{supplier4}} &= 14 \\
s_{\text{supplier5}} &= 19 \\
s_{\text{supplier6}} &= 70 \\
s_{\text{supplier7}} &= 60 \\
s_{\text{supplier8}} &= 39 \\
\end{align*}
\]

And
\[
x_{ij} \geq 0 \quad \forall i \in I, j \in J
\]

All indices, coefficients, and constraints are as retrieved and in source order.