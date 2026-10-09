##### Sets

- Warehouses (suppliers): $I = \{S1, S2, S3, S4, S5\}$
- Stores (customers): $J = \{D1, D2, D3, D4, D5\}$

##### Parameters

- Demand for each store $j$ ($d_j$):

| Store ($j$) | $d_j$ |
|:-----------:|:-----:|
| D1          | 428   |
| D2          | 217   |
| D3          | 214   |
| D4          | 380   |
| D5          | 254   |

- Supply capacity for each warehouse $i$ ($s_i$):

| Warehouse ($i$) | $s_i$ |
|:---------------:|:-----:|
| S1              | 428   |
| S2              | 217   |
| S3              | 214   |
| S4              | 380   |
| S5              | 254   |

- Transportation cost per unit from warehouse $i$ to store $j$ ($c_{ij}$):

|        | D1           | D2           | D3           | D4           | D5           |
|--------|--------------|--------------|--------------|--------------|--------------|
| S1     | 269.3910588  | 1.45373354   | 99.60345346  | 26.64078166  | 9.53768896   |
| S2     | 9.29184688   | 10.87477844  | 144.5260929  | 11.42013308  | 153.17568199 |
| S3     | 9.67458430   | 2.61916510   | 100.82422492 | 3.21219109   | 133.84933961 |
| S4     | 270.5749848  | 32.50253586  | 4.68420981   | 1.56822697   | 9.58927599   |
| S5     | 226.03319107 | 8.66916198   | 65.47681317  | 9.06876526   | 202.65015316 |

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous).

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each store must receive at least its demand.
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$

2. **Supply capacity:** Each warehouse cannot ship more than its supply capacity.
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$

3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Full Numerical Model

Let $x_{ij}$ be the quantity shipped from warehouse $i$ to store $j$.

Minimize:
\[
\begin{align*}
\min\ & 
269.3910588\,x_{S1,D1} + 1.45373354\,x_{S1,D2} + 99.60345346\,x_{S1,D3} + 26.64078166\,x_{S1,D4} + 9.53768896\,x_{S1,D5} \\
&+ 9.29184688\,x_{S2,D1} + 10.87477844\,x_{S2,D2} + 144.5260929\,x_{S2,D3} + 11.42013308\,x_{S2,D4} + 153.17568199\,x_{S2,D5} \\
&+ 9.67458430\,x_{S3,D1} + 2.61916510\,x_{S3,D2} + 100.82422492\,x_{S3,D3} + 3.21219109\,x_{S3,D4} + 133.84933961\,x_{S3,D5} \\
&+ 270.5749848\,x_{S4,D1} + 32.50253586\,x_{S4,D2} + 4.68420981\,x_{S4,D3} + 1.56822697\,x_{S4,D4} + 9.58927599\,x_{S4,D5} \\
&+ 226.03319107\,x_{S5,D1} + 8.66916198\,x_{S5,D2} + 65.47681317\,x_{S5,D3} + 9.06876526\,x_{S5,D4} + 202.65015316\,x_{S5,D5}
\end{align*}
\]

Subject to:

For each store:
\[
\begin{align*}
x_{S1,D1} + x_{S2,D1} + x_{S3,D1} + x_{S4,D1} + x_{S5,D1} &\geq 428 \\
x_{S1,D2} + x_{S2,D2} + x_{S3,D2} + x_{S4,D2} + x_{S5,D2} &\geq 217 \\
x_{S1,D3} + x_{S2,D3} + x_{S3,D3} + x_{S4,D3} + x_{S5,D3} &\geq 214 \\
x_{S1,D4} + x_{S2,D4} + x_{S3,D4} + x_{S4,D4} + x_{S5,D4} &\geq 380 \\
x_{S1,D5} + x_{S2,D5} + x_{S3,D5} + x_{S4,D5} + x_{S5,D5} &\geq 254 \\
\end{align*}
\]

For each warehouse:
\[
\begin{align*}
x_{S1,D1} + x_{S1,D2} + x_{S1,D3} + x_{S1,D4} + x_{S1,D5} &\leq 428 \\
x_{S2,D1} + x_{S2,D2} + x_{S2,D3} + x_{S2,D4} + x_{S2,D5} &\leq 217 \\
x_{S3,D1} + x_{S3,D2} + x_{S3,D3} + x_{S3,D4} + x_{S3,D5} &\leq 214 \\
x_{S4,D1} + x_{S4,D2} + x_{S4,D3} + x_{S4,D4} + x_{S4,D5} &\leq 380 \\
x_{S5,D1} + x_{S5,D2} + x_{S5,D3} + x_{S5,D4} + x_{S5,D5} &\leq 254 \\
\end{align*}
\]

Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in \{S1, S2, S3, S4, S5\},\ j \in \{D1, D2, D3, D4, D5\}
\]