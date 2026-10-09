##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

Where:
- $I = \{S1, S2, S3, S4, S5\}$ (warehouses)
- $J = \{D1, D2, D3, D4, D5\}$ (stores)

##### Parameters

- Store demands:
  - $d_{D1} = 428$
  - $d_{D2} = 217$
  - $d_{D3} = 214$
  - $d_{D4} = 380$
  - $d_{D5} = 254$

- Warehouse supply capacities:
  - $s_{S1} = 428$
  - $s_{S2} = 217$
  - $s_{S3} = 214$
  - $s_{S4} = 380$
  - $s_{S5} = 254$

- Transportation costs $c_{ij}$:

|        | D1                | D2                | D3                | D4                | D5                |
|--------|-------------------|-------------------|-------------------|-------------------|-------------------|
| S1     | 269.3910588020795 | 1.4537335390933939| 99.60345345756605 | 26.64078166309837 | 9.537688956880922 |
| S2     | 9.291846876785183 | 10.874778437070223| 144.52609291614627| 11.420133077898234| 153.1756819927813 |
| S3     | 9.674584301671008 | 2.6191650959687944| 100.8242249168735 | 3.2121910887916876| 133.8493396124168 |
| S4     | 270.57498480010247| 32.50253586       | 4.6842098096469815| 1.5682269686546804| 9.58927599        |
| S5     | 226.0331910675782 | 8.669161980826471 | 65.47681316968448 | 9.068765258459958 | 202.65015316425533|

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

That is,

\[
\begin{align*}
\min\ & 
269.3910588020795\,x_{S1,D1} + 1.4537335390933939\,x_{S1,D2} + 99.60345345756605\,x_{S1,D3} + 26.64078166309837\,x_{S1,D4} + 9.537688956880922\,x_{S1,D5} \\
&+ 9.291846876785183\,x_{S2,D1} + 10.874778437070223\,x_{S2,D2} + 144.52609291614627\,x_{S2,D3} + 11.420133077898234\,x_{S2,D4} + 153.1756819927813\,x_{S2,D5} \\
&+ 9.674584301671008\,x_{S3,D1} + 2.6191650959687944\,x_{S3,D2} + 100.8242249168735\,x_{S3,D3} + 3.2121910887916876\,x_{S3,D4} + 133.8493396124168\,x_{S3,D5} \\
&+ 270.57498480010247\,x_{S4,D1} + 32.50253586\,x_{S4,D2} + 4.6842098096469815\,x_{S4,D3} + 1.5682269686546804\,x_{S4,D4} + 9.58927599\,x_{S4,D5} \\
&+ 226.0331910675782\,x_{S5,D1} + 8.669161980826471\,x_{S5,D2} + 65.47681316968448\,x_{S5,D3} + 9.068765258459958\,x_{S5,D4} + 202.65015316425533\,x_{S5,D5}
\end{align*}
\]

##### Constraints

1. **Demand satisfaction (each store receives at least its demand):**

\[
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
\]

That is,

\[
\begin{align*}
x_{S1,D1} + x_{S2,D1} + x_{S3,D1} + x_{S4,D1} + x_{S5,D1} &\geq 428 \\
x_{S1,D2} + x_{S2,D2} + x_{S3,D2} + x_{S4,D2} + x_{S5,D2} &\geq 217 \\
x_{S1,D3} + x_{S2,D3} + x_{S3,D3} + x_{S4,D3} + x_{S5,D3} &\geq 214 \\
x_{S1,D4} + x_{S2,D4} + x_{S3,D4} + x_{S4,D4} + x_{S5,D4} &\geq 380 \\
x_{S1,D5} + x_{S2,D5} + x_{S3,D5} + x_{S4,D5} + x_{S5,D5} &\geq 254 \\
\end{align*}
\]

2. **Supply capacity (each warehouse ships no more than its capacity):**

\[
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
\]

That is,

\[
\begin{align*}
x_{S1,D1} + x_{S1,D2} + x_{S1,D3} + x_{S1,D4} + x_{S1,D5} &\leq 428 \\
x_{S2,D1} + x_{S2,D2} + x_{S2,D3} + x_{S2,D4} + x_{S2,D5} &\leq 217 \\
x_{S3,D1} + x_{S3,D2} + x_{S3,D3} + x_{S3,D4} + x_{S3,D5} &\leq 214 \\
x_{S4,D1} + x_{S4,D2} + x_{S4,D3} + x_{S4,D4} + x_{S4,D5} &\leq 380 \\
x_{S5,D1} + x_{S5,D2} + x_{S5,D3} + x_{S5,D4} + x_{S5,D5} &\leq 254 \\
\end{align*}
\]

3. **Non-negativity:**

\[
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\]

##### Summary

Minimize total transportation cost by choosing nonnegative shipment quantities $x_{ij}$ from each warehouse $i$ to each store $j$, so that all store demands are met, no warehouse exceeds its supply capacity, and all coefficients and identifiers are as above.