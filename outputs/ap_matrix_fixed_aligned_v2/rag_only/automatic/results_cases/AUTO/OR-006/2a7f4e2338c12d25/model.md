Sets:
- Warehouses: S1, S2, S3, S4, S5, S6, S7, S8, S9, S10
- Customers: C1, C2, C3, C4, C5, C6, C7, C8, C9, C10

Parameters (from source order):

Customer Demands (customer_demand.csv):
- demand_C1 = 45
- demand_C2 = 23
- demand_C3 = 94
- demand_C4 = 92
- demand_C5 = 57
- demand_C6 = 52
- demand_C7 = 23
- demand_C8 = 99
- demand_C9 = 99
- demand_C10 = 77

Warehouse Supply Capacities (supply_capacity.csv):
- supply_S1 = 127
- supply_S2 = 236
- supply_S3 = 168
- supply_S4 = 115
- supply_S5 = 280
- supply_S6 = 179
- supply_S7 = 135
- supply_S8 = 263
- supply_S9 = 283
- supply_S10 = 476

Transportation Costs (transportation_costs.csv, cost_Si_Cj):

|        |  C1         |  C2         |  C3         |  C4         |  C5         |  C6         |  C7         |  C8         |  C9         |  C10        |
|--------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| S1     | 2077.058672521021 | 0.0         | 54.33526480458508 | 0.0         | 0.0         | 36.17284629162332 | 0.0         | 0.0         | 169.33026926588778 | 0.0         |
| S2     | 2077.058672521021 | 0.0         | 1141.0405608962865 | 0.0         | 0.0         | 651.1112332492198 | 0.0         | 0.0         | 8.063346155518467 | 0.0         |
| S3     | 79.9210295982608 | 474.24509131006675 | 1477.0676289106607 | 22.583099586193658 | 474.24509131006675 | 41.106596962251096 | 474.24509131006675 | 474.24509131006675 | 624.162539502301 | 474.24509131006675 |
| S4     | 1659.336929105112 | 57.20541468776147 | 186.1519048103841 | 1201.3137084429907 | 1029.6974643797064 | 41.82210594495074 | 57.20541468776147 | 1201.3137084429907 | 884.5633870657458 | 1029.6974643797064 |
| S5     | 1297.2567040858307 | 77.76629131320436 | 24.26760227579214 | 1399.7932436376784 | 77.76629131320436 | 53.91161728496604 | 1399.7932436376784 | 77.76629131320436 | 1255.115148013589 | 1399.7932436376784 |
| S6     | 1998.9090658724567 | 985.3165435695341 | 2.8541686885814643 | 1149.53596749779 | 985.3165435695341 | 730.6923647662475 | 54.73980797608523 | 985.3165435695341 | 46.803102206463265 | 1149.53596749779 |
| S7     | 1780.3360050180179 | 0.0         | 1141.0405608962865 | 0.0         | 0.0         | 36.17284629162332 | 0.0         | 0.0         | 8.063346155518467 | 0.0         |
| S8     | 75.40935896233042 | 1338.1987290721909 | 21.391345987195333 | 74.34437383734394 | 74.34437383734394 | 937.3506239061463 | 1338.1987290721909 | 1338.1987290721909 | 1392.1186581383768 | 1338.1987290721909 |
| S9     | 98.90755583433433 | 0.0         | 978.0347664825314 | 0.0         | 0.0         | 651.1112332492198 | 0.0         | 0.0         | 169.33026926588778 | 0.0         |
| S10    | 2077.058672521021 | 0.0         | 54.33526480458508 | 0.0         | 0.0         | 36.17284629162332 | 0.0         | 0.0         | 145.1402307993324 | 0.0         |

Decision Variables:
- x_{i,j} = quantity shipped from warehouse Si to customer Cj, for all i in {1,...,10}, j in {1,...,10}
- Domain: x_{i,j} ≥ 0, continuous

Mathematical Formulation:

Minimize total transportation cost:
\[
\text{Minimize} \quad Z = 
\sum_{i=1}^{10} \sum_{j=1}^{10} \text{cost}_{S_i,C_j} \cdot x_{i,j}
\]
where cost_{S_i,C_j} is as given in the table above.

Subject to:

1. Demand satisfaction for each customer (from customer_demand.csv, in source order):
\[
\begin{align*}
\sum_{i=1}^{10} x_{i,1} &= 45 \\
\sum_{i=1}^{10} x_{i,2} &= 23 \\
\sum_{i=1}^{10} x_{i,3} &= 94 \\
\sum_{i=1}^{10} x_{i,4} &= 92 \\
\sum_{i=1}^{10} x_{i,5} &= 57 \\
\sum_{i=1}^{10} x_{i,6} &= 52 \\
\sum_{i=1}^{10} x_{i,7} &= 23 \\
\sum_{i=1}^{10} x_{i,8} &= 99 \\
\sum_{i=1}^{10} x_{i,9} &= 99 \\
\sum_{i=1}^{10} x_{i,10} &= 77 \\
\end{align*}
\]

2. Supply capacity for each warehouse (from supply_capacity.csv, in source order):
\[
\begin{align*}
\sum_{j=1}^{10} x_{1,j} &\leq 127 \\
\sum_{j=1}^{10} x_{2,j} &\leq 236 \\
\sum_{j=1}^{10} x_{3,j} &\leq 168 \\
\sum_{j=1}^{10} x_{4,j} &\leq 115 \\
\sum_{j=1}^{10} x_{5,j} &\leq 280 \\
\sum_{j=1}^{10} x_{6,j} &\leq 179 \\
\sum_{j=1}^{10} x_{7,j} &\leq 135 \\
\sum_{j=1}^{10} x_{8,j} &\leq 263 \\
\sum_{j=1}^{10} x_{9,j} &\leq 283 \\
\sum_{j=1}^{10} x_{10,j} &\leq 476 \\
\end{align*}
\]

3. Non-negativity:
\[
x_{i,j} \geq 0 \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,10\}
\]

All coefficients and identifiers are preserved in source order as required. No additional constraints or modifications have been introduced.

This is the complete numerical formulation of the transportation problem as described.