##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$ (warehouses), $j \in J$ (stores).

##### Sets

- Warehouses $I = \{S1, S2, S3, S4, S5\}$
- Stores $J = \{D1, D2, D3, D4, D5\}$

##### Parameters

- Store demands (units):

  - $d_{D1} = 428$
  - $d_{D2} = 217$
  - $d_{D3} = 214$
  - $d_{D4} = 380$
  - $d_{D5} = 254$

- Warehouse supply capacities (units):

  - $s_{S1} = 428$
  - $s_{S2} = 217$
  - $s_{S3} = 214$
  - $s_{S4} = 380$
  - $s_{S5} = 254$

- Transportation costs per unit ($c_{ij}$):

  |         | D1              | D2              | D3              | D4              | D5              |
  |---------|-----------------|-----------------|-----------------|-----------------|-----------------|
  | S1      | 269.39105880208 | 1.4537335390934 | 99.603453457566 | 26.640781663098 | 9.5376889568809 |
  | S2      | 9.2918468767852 | 10.874778437070 | 144.52609291615 | 11.420133077898 | 153.17568199278 |
  | S3      | 9.6745843016710 | 2.6191650959688 | 100.82422491687 | 3.2121910887917 | 133.84933961242 |
  | S4      | 270.57498480010 | 32.50253586     | 4.6842098096470 | 1.5682269686547 | 9.58927599      |
  | S5      | 226.03319106758 | 8.6691619808265 | 65.476813169684 | 9.0687652584600 | 202.65015316426 |

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each store must receive at least its demand.
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
   Explicitly:
   \begin{align*}
   x_{S1,D1} + x_{S2,D1} + x_{S3,D1} + x_{S4,D1} + x_{S5,D1} &\geq 428 \\
   x_{S1,D2} + x_{S2,D2} + x_{S3,D2} + x_{S4,D2} + x_{S5,D2} &\geq 217 \\
   x_{S1,D3} + x_{S2,D3} + x_{S3,D3} + x_{S4,D3} + x_{S5,D3} &\geq 214 \\
   x_{S1,D4} + x_{S2,D4} + x_{S3,D4} + x_{S4,D4} + x_{S5,D4} &\geq 380 \\
   x_{S1,D5} + x_{S2,D5} + x_{S3,D5} + x_{S4,D5} + x_{S5,D5} &\geq 254 \\
   \end{align*}

2. **Supply capacity:** Each warehouse cannot ship more than its capacity.
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
   Explicitly:
   \begin{align*}
   x_{S1,D1} + x_{S1,D2} + x_{S1,D3} + x_{S1,D4} + x_{S1,D5} &\leq 428 \\
   x_{S2,D1} + x_{S2,D2} + x_{S2,D3} + x_{S2,D4} + x_{S2,D5} &\leq 217 \\
   x_{S3,D1} + x_{S3,D2} + x_{S3,D3} + x_{S3,D4} + x_{S3,D5} &\leq 214 \\
   x_{S4,D1} + x_{S4,D2} + x_{S4,D3} + x_{S4,D4} + x_{S4,D5} &\leq 380 \\
   x_{S5,D1} + x_{S5,D2} + x_{S5,D3} + x_{S5,D4} + x_{S5,D5} &\leq 254 \\
   \end{align*}

3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Complete Model

Minimize
$$
\begin{align*}
&269.39105880208\,x_{S1,D1} + 1.4537335390934\,x_{S1,D2} + 99.603453457566\,x_{S1,D3} + 26.640781663098\,x_{S1,D4} + 9.5376889568809\,x_{S1,D5} \\
+&9.2918468767852\,x_{S2,D1} + 10.874778437070\,x_{S2,D2} + 144.52609291615\,x_{S2,D3} + 11.420133077898\,x_{S2,D4} + 153.17568199278\,x_{S2,D5} \\
+&9.6745843016710\,x_{S3,D1} + 2.6191650959688\,x_{S3,D2} + 100.82422491687\,x_{S3,D3} + 3.2121910887917\,x_{S3,D4} + 133.84933961242\,x_{S3,D5} \\
+&270.57498480010\,x_{S4,D1} + 32.50253586\,x_{S4,D2} + 4.6842098096470\,x_{S4,D3} + 1.5682269686547\,x_{S4,D4} + 9.58927599\,x_{S4,D5} \\
+&226.03319106758\,x_{S5,D1} + 8.6691619808265\,x_{S5,D2} + 65.476813169684\,x_{S5,D3} + 9.0687652584600\,x_{S5,D4} + 202.65015316426\,x_{S5,D5}
\end{align*}
$$

Subject to:

\[
\begin{align*}
x_{S1,D1} + x_{S2,D1} + x_{S3,D1} + x_{S4,D1} + x_{S5,D1} &\geq 428 \\
x_{S1,D2} + x_{S2,D2} + x_{S3,D2} + x_{S4,D2} + x_{S5,D2} &\geq 217 \\
x_{S1,D3} + x_{S2,D3} + x_{S3,D3} + x_{S4,D3} + x_{S5,D3} &\geq 214 \\
x_{S1,D4} + x_{S2,D4} + x_{S3,D4} + x_{S4,D4} + x_{S5,D4} &\geq 380 \\
x_{S1,D5} + x_{S2,D5} + x_{S3,D5} + x_{S4,D5} + x_{S5,D5} &\geq 254 \\
x_{S1,D1} + x_{S1,D2} + x_{S1,D3} + x_{S1,D4} + x_{S1,D5} &\leq 428 \\
x_{S2,D1} + x_{S2,D2} + x_{S2,D3} + x_{S2,D4} + x_{S2,D5} &\leq 217 \\
x_{S3,D1} + x_{S3,D2} + x_{S3,D3} + x_{S3,D4} + x_{S3,D5} &\leq 214 \\
x_{S4,D1} + x_{S4,D2} + x_{S4,D3} + x_{S4,D4} + x_{S4,D5} &\leq 380 \\
x_{S5,D1} + x_{S5,D2} + x_{S5,D3} + x_{S5,D4} + x_{S5,D5} &\leq 254 \\
x_{ij} &\geq 0 \quad \forall i \in I,\, j \in J
\end{align*}
\]