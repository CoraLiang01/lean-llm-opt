Let:
- \( S = \{S1, S2, S3, S4, S5\} \) be the set of warehouses (supply regions).
- \( D = \{D1, D2, D3, D4, D5\} \) be the set of stores (demand points).
- \( x_{ij} \) = quantity shipped from warehouse \( i \in S \) to store \( j \in D \), for all \( i, j \).

Parameters (from CSVs, source order preserved):

Store demands (customer_demand.csv):
- \( \text{Demand}_{D1} = 428 \)
- \( \text{Demand}_{D2} = 217 \)
- \( \text{Demand}_{D3} = 214 \)
- \( \text{Demand}_{D4} = 380 \)
- \( \text{Demand}_{D5} = 254 \)

Warehouse supply capacities (supply_capacity.csv):
- \( \text{Supply}_{S1} = 428 \)
- \( \text{Supply}_{S2} = 217 \)
- \( \text{Supply}_{S3} = 214 \)
- \( \text{Supply}_{S4} = 380 \)
- \( \text{Supply}_{S5} = 254 \)

Transportation costs per unit (transportation_costs.csv):

|        | D1              | D2              | D3              | D4              | D5              |
|--------|-----------------|-----------------|-----------------|-----------------|-----------------|
| S1     | 269.39105880208 | 1.45373353909   | 99.60345345757  | 26.64078166310  | 9.53768895688   |
| S2     | 9.29184687679   | 10.87477843707  | 144.52609291615 | 11.42013307790  | 153.17568199278 |
| S3     | 9.67458430167   | 2.61916509597   | 100.82422491687 | 3.21219108879   | 133.84933961242 |
| S4     | 270.57498480010 | 32.50253586     | 4.68420980965   | 1.56822696865   | 9.58927599      |
| S5     | 226.03319106758 | 8.66916198083   | 65.47681316968  | 9.06876525846   | 202.65015316426 |

Decision variables:
- \( x_{ij} \geq 0 \), continuous, for all \( i \in S, j \in D \).

Objective:
Minimize total transportation cost:
\[
\begin{align*}
\min \quad & 269.39105880208\,x_{S1,D1} + 1.45373353909\,x_{S1,D2} + 99.60345345757\,x_{S1,D3} + 26.64078166310\,x_{S1,D4} + 9.53768895688\,x_{S1,D5} \\
&+ 9.29184687679\,x_{S2,D1} + 10.87477843707\,x_{S2,D2} + 144.52609291615\,x_{S2,D3} + 11.42013307790\,x_{S2,D4} + 153.17568199278\,x_{S2,D5} \\
&+ 9.67458430167\,x_{S3,D1} + 2.61916509597\,x_{S3,D2} + 100.82422491687\,x_{S3,D3} + 3.21219108879\,x_{S3,D4} + 133.84933961242\,x_{S3,D5} \\
&+ 270.57498480010\,x_{S4,D1} + 32.50253586\,x_{S4,D2} + 4.68420980965\,x_{S4,D3} + 1.56822696865\,x_{S4,D4} + 9.58927599\,x_{S4,D5} \\
&+ 226.03319106758\,x_{S5,D1} + 8.66916198083\,x_{S5,D2} + 65.47681316968\,x_{S5,D3} + 9.06876525846\,x_{S5,D4} + 202.65015316426\,x_{S5,D5}
\end{align*}
\]

Subject to:

1. Demand satisfaction for each store (customer):
\[
\begin{align*}
x_{S1,D1} + x_{S2,D1} + x_{S3,D1} + x_{S4,D1} + x_{S5,D1} &= 428 \\
x_{S1,D2} + x_{S2,D2} + x_{S3,D2} + x_{S4,D2} + x_{S5,D2} &= 217 \\
x_{S1,D3} + x_{S2,D3} + x_{S3,D3} + x_{S4,D3} + x_{S5,D3} &= 214 \\
x_{S1,D4} + x_{S2,D4} + x_{S3,D4} + x_{S4,D4} + x_{S5,D4} &= 380 \\
x_{S1,D5} + x_{S2,D5} + x_{S3,D5} + x_{S4,D5} + x_{S5,D5} &= 254 \\
\end{align*}
\]

2. Supply capacity for each warehouse (region):
\[
\begin{align*}
x_{S1,D1} + x_{S1,D2} + x_{S1,D3} + x_{S1,D4} + x_{S1,D5} &\leq 428 \\
x_{S2,D1} + x_{S2,D2} + x_{S2,D3} + x_{S2,D4} + x_{S2,D5} &\leq 217 \\
x_{S3,D1} + x_{S3,D2} + x_{S3,D3} + x_{S3,D4} + x_{S3,D5} &\leq 214 \\
x_{S4,D1} + x_{S4,D2} + x_{S4,D3} + x_{S4,D4} + x_{S4,D5} &\leq 380 \\
x_{S5,D1} + x_{S5,D2} + x_{S5,D3} + x_{S5,D4} + x_{S5,D5} &\leq 254 \\
\end{align*}
\]

3. Nonnegativity:
\[
x_{ij} \geq 0 \quad \forall i \in S, j \in D
\]

This is a complete numerical linear programming formulation for the GreenMart transportation problem, preserving all source-ordered data and semantics.