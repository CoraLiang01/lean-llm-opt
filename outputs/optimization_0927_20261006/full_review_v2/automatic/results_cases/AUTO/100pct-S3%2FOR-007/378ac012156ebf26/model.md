Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following ProductName values in the order retrieved:

1. Sedan
2. SUV
3. Truck
4. Convertible
5. Minivan
6. Coupe
7. Hatchback
8. Station Wagon
9. Electric Car
10. Hybrid Car
11. Luxury Sedan
12. Sports Car
13. Crossover
14. Diesel Truck
15. Compact SUV
16. Luxury SUV
17. Cargo Van
18. Pickup Truck
19. Roadster
20. Muscle Car
21. Off-road Vehicle
22. Camper Van
23. Compact Car
24. Motorcycle
25. Electric SUV

Let $p_i$ be the profit (Value) for each vehicle type $i$, and $w_i$ be the Weight (stock space requirement) for each vehicle type $i$. The total available inventory capacity is $765$.

The model is:

Objective:
\[
\max \sum_{i=1}^{25} p_i x_i
\]

where

\[
\begin{align*}
p_1 &= 2524 & w_1 &= 99 \\
p_2 &= 4614 & w_2 &= 55 \\
p_3 &= 8416 & w_3 &= 75 \\
p_4 &= 5917 & w_4 &= 94 \\
p_5 &= 9048 & w_5 &= 80 \\
p_6 &= 1140 & w_6 &= 82 \\
p_7 &= 8962 & w_7 &= 71 \\
p_8 &= 1888 & w_8 &= 100 \\
p_9 &= 8487 & w_9 &= 28 \\
p_{10} &= 4425 & w_{10} &= 93 \\
p_{11} &= 4717 & w_{11} &= 84 \\
p_{12} &= 4210 & w_{12} &= 83 \\
p_{13} &= 1226 & w_{13} &= 62 \\
p_{14} &= 7400 & w_{14} &= 90 \\
p_{15} &= 4639 & w_{15} &= 99 \\
p_{16} &= 7712 & w_{16} &= 96 \\
p_{17} &= 3299 & w_{17} &= 21 \\
p_{18} &= 9895 & w_{18} &= 39 \\
p_{19} &= 4496 & w_{19} &= 99 \\
p_{20} &= 4526 & w_{20} &= 81 \\
p_{21} &= 5688 & w_{21} &= 6 \\
p_{22} &= 3007 & w_{22} &= 58 \\
p_{23} &= 3623 & w_{23} &= 37 \\
p_{24} &= 8474 & w_{24} &= 15 \\
p_{25} &= 8372 & w_{25} &= 37 \\
\end{align*}
\]

Subject to:

Capacity constraint:
\[
\sum_{i=1}^{25} w_i x_i \leq 765
\]

Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 25
\]

Where:
- $x_i$ = number of vehicles of type $i$ to order per day (integer, $\geq 0$)
- $p_i$ = profit per unit of vehicle type $i$ (from Value column)
- $w_i$ = stock space required per unit of vehicle type $i$ (from Weight column)
- Total inventory capacity = $765$

All coefficients and identifiers are as retrieved and in original order.