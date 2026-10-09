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

Let $p_i$ be the Value (profit) for each vehicle type $i$, and $w_i$ be the Weight (stock space required) for each vehicle type $i$. The total available inventory capacity is $765$.

The model is:

$$
\begin{align*}
\max \quad & 2524 x_1 + 4614 x_2 + 8416 x_3 + 5917 x_4 + 9048 x_5 + 1140 x_6 + 8962 x_7 + 1888 x_8 + 8487 x_9 + 4425 x_{10} \\
& + 4717 x_{11} + 4210 x_{12} + 1226 x_{13} + 7400 x_{14} + 4639 x_{15} + 7712 x_{16} + 3299 x_{17} + 9895 x_{18} \\
& + 4496 x_{19} + 4526 x_{20} + 5688 x_{21} + 3007 x_{22} + 3623 x_{23} + 8474 x_{24} + 8372 x_{25} \\
\text{s.t.} \quad & 99 x_1 + 55 x_2 + 75 x_3 + 94 x_4 + 80 x_5 + 82 x_6 + 71 x_7 + 100 x_8 + 28 x_9 + 93 x_{10} \\
& + 84 x_{11} + 83 x_{12} + 62 x_{13} + 90 x_{14} + 99 x_{15} + 96 x_{16} + 21 x_{17} + 39 x_{18} \\
& + 99 x_{19} + 81 x_{20} + 6 x_{21} + 58 x_{22} + 37 x_{23} + 15 x_{24} + 37 x_{25} \leq 765 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 25
\end{align*}
$$

Where:

- $x_i$ = number of vehicles of type $i$ to order per day (nonnegative integer)
- $p_i$ = Value (profit) for vehicle type $i$ (see coefficients above)
- $w_i$ = Weight (stock space required) for vehicle type $i$ (see coefficients above)
- Total inventory capacity = $765$

All data and coefficients are as retrieved and in original order.