##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$ (continuous).

##### Parameters

- Warehouses (regions): $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- Stores (customers): $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

- Demands:
  - $d_{\text{D1}} = 428$
  - $d_{\text{D2}} = 217$
  - $d_{\text{D3}} = 214$
  - $d_{\text{D4}} = 380$
  - $d_{\text{D5}} = 254$

- Supply capacities:
  - $s_{\text{S1}} = 428$
  - $s_{\text{S2}} = 217$
  - $s_{\text{S3}} = 214$
  - $s_{\text{S4}} = 380$
  - $s_{\text{S5}} = 254$

- Transportation costs $c_{ij}$:

\[
\begin{array}{c|ccccc}
      & \text{D1} & \text{D2} & \text{D3} & \text{D4} & \text{D5} \\
\hline
\text{S1} & 269.3910588020795 & 1.453733539093394 & 99.60345345756603 & 26.64078166309837 & 9.537688956880922 \\
\text{S2} & 9.291846876785185 & 10.874778437070225 & 144.52609291614627 & 11.420133077898234 & 153.1756819927813 \\
\text{S3} & 9.674584301671008 & 2.6191650959687944 & 100.8242249168735 & 3.212191088791688 & 133.8493396124168 \\
\text{S4} & 270.57498480010247 & 32.50253586 & 4.6842098096469815 & 1.5682269686546804 & 9.58927599 \\
\text{S5} & 226.0331910675782 & 8.669161980826471 & 65.47681316968448 & 9.068765258459958 & 202.65015316425533 \\
\end{array}
\]

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

That is,

\[
\begin{align*}
\min\ & 
269.3910588020795\,x_{\text{S1},\text{D1}} + 1.453733539093394\,x_{\text{S1},\text{D2}} + 99.60345345756603\,x_{\text{S1},\text{D3}} + 26.64078166309837\,x_{\text{S1},\text{D4}} + 9.537688956880922\,x_{\text{S1},\text{D5}} \\
&+ 9.291846876785185\,x_{\text{S2},\text{D1}} + 10.874778437070225\,x_{\text{S2},\text{D2}} + 144.52609291614627\,x_{\text{S2},\text{D3}} + 11.420133077898234\,x_{\text{S2},\text{D4}} + 153.1756819927813\,x_{\text{S2},\text{D5}} \\
&+ 9.674584301671008\,x_{\text{S3},\text{D1}} + 2.6191650959687944\,x_{\text{S3},\text{D2}} + 100.8242249168735\,x_{\text{S3},\text{D3}} + 3.212191088791688\,x_{\text{S3},\text{D4}} + 133.8493396124168\,x_{\text{S3},\text{D5}} \\
&+ 270.57498480010247\,x_{\text{S4},\text{D1}} + 32.50253586\,x_{\text{S4},\text{D2}} + 4.6842098096469815\,x_{\text{S4},\text{D3}} + 1.5682269686546804\,x_{\text{S4},\text{D4}} + 9.58927599\,x_{\text{S4},\text{D5}} \\
&+ 226.0331910675782\,x_{\text{S5},\text{D1}} + 8.669161980826471\,x_{\text{S5},\text{D2}} + 65.47681316968448\,x_{\text{S5},\text{D3}} + 9.068765258459958\,x_{\text{S5},\text{D4}} + 202.65015316425533\,x_{\text{S5},\text{D5}}
\end{align*}
\]

##### Constraints

1. Demand satisfaction (for each store $j$):

\[
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
\]

That is,

\[
\begin{align*}
x_{\text{S1},\text{D1}} + x_{\text{S2},\text{D1}} + x_{\text{S3},\text{D1}} + x_{\text{S4},\text{D1}} + x_{\text{S5},\text{D1}} &\geq 428 \\
x_{\text{S1},\text{D2}} + x_{\text{S2},\text{D2}} + x_{\text{S3},\text{D2}} + x_{\text{S4},\text{D2}} + x_{\text{S5},\text{D2}} &\geq 217 \\
x_{\text{S1},\text{D3}} + x_{\text{S2},\text{D3}} + x_{\text{S3},\text{D3}} + x_{\text{S4},\text{D3}} + x_{\text{S5},\text{D3}} &\geq 214 \\
x_{\text{S1},\text{D4}} + x_{\text{S2},\text{D4}} + x_{\text{S3},\text{D4}} + x_{\text{S4},\text{D4}} + x_{\text{S5},\text{D4}} &\geq 380 \\
x_{\text{S1},\text{D5}} + x_{\text{S2},\text{D5}} + x_{\text{S3},\text{D5}} + x_{\text{S4},\text{D5}} + x_{\text{S5},\text{D5}} &\geq 254 \\
\end{align*}
\]

2. Supply capacity (for each warehouse $i$):

\[
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
\]

That is,

\[
\begin{align*}
x_{\text{S1},\text{D1}} + x_{\text{S1},\text{D2}} + x_{\text{S1},\text{D3}} + x_{\text{S1},\text{D4}} + x_{\text{S1},\text{D5}} &\leq 428 \\
x_{\text{S2},\text{D1}} + x_{\text{S2},\text{D2}} + x_{\text{S2},\text{D3}} + x_{\text{S2},\text{D4}} + x_{\text{S2},\text{D5}} &\leq 217 \\
x_{\text{S3},\text{D1}} + x_{\text{S3},\text{D2}} + x_{\text{S3},\text{D3}} + x_{\text{S3},\text{D4}} + x_{\text{S3},\text{D5}} &\leq 214 \\
x_{\text{S4},\text{D1}} + x_{\text{S4},\text{D2}} + x_{\text{S4},\text{D3}} + x_{\text{S4},\text{D4}} + x_{\text{S4},\text{D5}} &\leq 380 \\
x_{\text{S5},\text{D1}} + x_{\text{S5},\text{D2}} + x_{\text{S5},\text{D3}} + x_{\text{S5},\text{D4}} + x_{\text{S5},\text{D5}} &\leq 254 \\
\end{align*}
\]

3. Non-negativity:

\[
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\]