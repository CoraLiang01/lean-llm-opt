**Index Set:**  
Let  
$\mathcal{I} = \{$  
 "S700_1138",  
 "S700_1691",  
 "S700_1938",  
 "S700_2047",  
 "S700_2466",  
 "S700_2610",  
 "S700_2824",  
 "S700_2834",  
 "S700_3167",  
 "S700_3505",  
 "S700_3962",  
 "S700_4002"  
$\}$  
(in source order).

**Parameters:**  
For each $i \in \mathcal{I}$:

| Product Name      | $A_i$ (Revenue) | $d_i$ (Demand) | $I_i$ (Initial Inventory) |
|-------------------|-----------------|----------------|--------------------------|
| S700_1138         | 70.67           | 1219           | 9020                     |
| S700_1691         | 100.0           | 1127           | 8370                     |
| S700_1938         | 70.15           | 1129           | 8390                     |
| S700_2047         | 100.0           | 1176           | 8680                     |
| S700_2466         | 100.0           | 1301           | 9400                     |
| S700_2610         | 65.77           | 1340           | 9900                     |
| S700_2824         | 100.0           | 1357           | 9760                     |
| S700_2834         | 100.0           | 1158           | 8610                     |
| S700_3167         | 74.4            | 1287           | 9380                     |
| S700_3505         | 81.14           | 1281           | 9170                     |
| S700_3962         | 100.0           | 1135           | 8520                     |
| S700_4002         | 61.44           | 1392           | 10290                    |

**Decision Variables:**  
For each $i \in \mathcal{I}$:  
$x_i \in \mathbb{Z}_+, \quad$ number of units of product $i$ to fulfill.

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

**Constraints:**  
For each $i \in \mathcal{I}$:
1. Inventory constraint:
$$
x_i \leq I_i
$$
2. Demand constraint:
$$
x_i \leq d_i
$$
3. Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
$$

**Explicitly, for each product:**

For $i$ = S700_1138:
- $x_{\text{S700\_1138}} \leq 9020$
- $x_{\text{S700\_1138}} \leq 1219$
- $x_{\text{S700\_1138}} \in \mathbb{Z}_+$

For $i$ = S700_1691:
- $x_{\text{S700\_1691}} \leq 8370$
- $x_{\text{S700\_1691}} \leq 1127$
- $x_{\text{S700\_1691}} \in \mathbb{Z}_+$

For $i$ = S700_1938:
- $x_{\text{S700\_1938}} \leq 8390$
- $x_{\text{S700\_1938}} \leq 1129$
- $x_{\text{S700\_1938}} \in \mathbb{Z}_+$

For $i$ = S700_2047:
- $x_{\text{S700\_2047}} \leq 8680$
- $x_{\text{S700\_2047}} \leq 1176$
- $x_{\text{S700\_2047}} \in \mathbb{Z}_+$

For $i$ = S700_2466:
- $x_{\text{S700\_2466}} \leq 9400$
- $x_{\text{S700\_2466}} \leq 1301$
- $x_{\text{S700\_2466}} \in \mathbb{Z}_+$

For $i$ = S700_2610:
- $x_{\text{S700\_2610}} \leq 9900$
- $x_{\text{S700\_2610}} \leq 1340$
- $x_{\text{S700\_2610}} \in \mathbb{Z}_+$

For $i$ = S700_2824:
- $x_{\text{S700\_2824}} \leq 9760$
- $x_{\text{S700\_2824}} \leq 1357$
- $x_{\text{S700\_2824}} \in \mathbb{Z}_+$

For $i$ = S700_2834:
- $x_{\text{S700\_2834}} \leq 8610$
- $x_{\text{S700\_2834}} \leq 1158$
- $x_{\text{S700\_2834}} \in \mathbb{Z}_+$

For $i$ = S700_3167:
- $x_{\text{S700\_3167}} \leq 9380$
- $x_{\text{S700\_3167}} \leq 1287$
- $x_{\text{S700\_3167}} \in \mathbb{Z}_+$

For $i$ = S700_3505:
- $x_{\text{S700\_3505}} \leq 9170$
- $x_{\text{S700\_3505}} \leq 1281$
- $x_{\text{S700\_3505}} \in \mathbb{Z}_+$

For $i$ = S700_3962:
- $x_{\text{S700\_3962}} \leq 8520$
- $x_{\text{S700\_3962}} \leq 1135$
- $x_{\text{S700\_3962}} \in \mathbb{Z}_+$

For $i$ = S700_4002:
- $x_{\text{S700\_4002}} \leq 10290$
- $x_{\text{S700\_4002}} \leq 1392$
- $x_{\text{S700\_4002}} \in \mathbb{Z}_+$

**Summary Table of Parameters (in source order):**

| $i$                | $A_i$   | $d_i$ | $I_i$  |
|--------------------|---------|-------|--------|
| S700_1138          | 70.67   | 1219  | 9020   |
| S700_1691          | 100.0   | 1127  | 8370   |
| S700_1938          | 70.15   | 1129  | 8390   |
| S700_2047          | 100.0   | 1176  | 8680   |
| S700_2466          | 100.0   | 1301  | 9400   |
| S700_2610          | 65.77   | 1340  | 9900   |
| S700_2824          | 100.0   | 1357  | 9760   |
| S700_2834          | 100.0   | 1158  | 8610   |
| S700_3167          | 74.4    | 1287  | 9380   |
| S700_3505          | 81.14   | 1281  | 9170   |
| S700_3962          | 100.0   | 1135  | 8520   |
| S700_4002          | 61.44   | 1392  | 10290  |

**Complete Model:**

$$
\begin{align*}
\max \quad & 70.67\, x_{\text{S700\_1138}} + 100.0\, x_{\text{S700\_1691}} + 70.15\, x_{\text{S700\_1938}} + 100.0\, x_{\text{S700\_2047}} \\
& + 100.0\, x_{\text{S700\_2466}} + 65.77\, x_{\text{S700\_2610}} + 100.0\, x_{\text{S700\_2824}} + 100.0\, x_{\text{S700\_2834}} \\
& + 74.4\, x_{\text{S700\_3167}} + 81.14\, x_{\text{S700\_3505}} + 100.0\, x_{\text{S700\_3962}} + 61.44\, x_{\text{S700\_4002}} \\
\text{s.t.} \quad & 0 \leq x_{\text{S700\_1138}} \leq \min\{1219, 9020\} \\
& 0 \leq x_{\text{S700\_1691}} \leq \min\{1127, 8370\} \\
& 0 \leq x_{\text{S700\_1938}} \leq \min\{1129, 8390\} \\
& 0 \leq x_{\text{S700\_2047}} \leq \min\{1176, 8680\} \\
& 0 \leq x_{\text{S700\_2466}} \leq \min\{1301, 9400\} \\
& 0 \leq x_{\text{S700\_2610}} \leq \min\{1340, 9900\} \\
& 0 \leq x_{\text{S700\_2824}} \leq \min\{1357, 9760\} \\
& 0 \leq x_{\text{S700\_2834}} \leq \min\{1158, 8610\} \\
& 0 \leq x_{\text{S700\_3167}} \leq \min\{1287, 9380\} \\
& 0 \leq x_{\text{S700\_3505}} \leq \min\{1281, 9170\} \\
& 0 \leq x_{\text{S700\_3962}} \leq \min\{1135, 8520\} \\
& 0 \leq x_{\text{S700\_4002}} \leq \min\{1392, 10290\} \\
& x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
\end{align*}
$$

where each $x_i$ is the number of units of product $i$ to fulfill.