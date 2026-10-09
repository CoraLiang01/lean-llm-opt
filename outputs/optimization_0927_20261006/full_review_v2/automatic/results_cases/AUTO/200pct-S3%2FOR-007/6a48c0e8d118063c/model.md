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

Let $p_i$ be the profit (Value) for each vehicle type $i$, and $w_i$ be the weight (Weight) for each vehicle type $i$. The total inventory capacity is $765$.

##### Objective Function

\[
\max \sum_{i=1}^{25} p_i x_i
\]

where

\[
\begin{align*}
p_1 &= 2524 & \text{(Sedan)} \\
p_2 &= 4614 & \text{(SUV)} \\
p_3 &= 8416 & \text{(Truck)} \\
p_4 &= 5917 & \text{(Convertible)} \\
p_5 &= 9048 & \text{(Minivan)} \\
p_6 &= 1140 & \text{(Coupe)} \\
p_7 &= 8962 & \text{(Hatchback)} \\
p_8 &= 1888 & \text{(Station Wagon)} \\
p_9 &= 8487 & \text{(Electric Car)} \\
p_{10} &= 4425 & \text{(Hybrid Car)} \\
p_{11} &= 4717 & \text{(Luxury Sedan)} \\
p_{12} &= 4210 & \text{(Sports Car)} \\
p_{13} &= 1226 & \text{(Crossover)} \\
p_{14} &= 7400 & \text{(Diesel Truck)} \\
p_{15} &= 4639 & \text{(Compact SUV)} \\
p_{16} &= 7712 & \text{(Luxury SUV)} \\
p_{17} &= 3299 & \text{(Cargo Van)} \\
p_{18} &= 9895 & \text{(Pickup Truck)} \\
p_{19} &= 4496 & \text{(Roadster)} \\
p_{20} &= 4526 & \text{(Muscle Car)} \\
p_{21} &= 5688 & \text{(Off-road Vehicle)} \\
p_{22} &= 3007 & \text{(Camper Van)} \\
p_{23} &= 3623 & \text{(Compact Car)} \\
p_{24} &= 8474 & \text{(Motorcycle)} \\
p_{25} &= 8372 & \text{(Electric SUV)} \\
\end{align*}
\]

##### Constraints

**Inventory Capacity Constraint:**

\[
\sum_{i=1}^{25} w_i x_i \leq 765
\]

where

\[
\begin{align*}
w_1 &= 99 & \text{(Sedan)} \\
w_2 &= 55 & \text{(SUV)} \\
w_3 &= 75 & \text{(Truck)} \\
w_4 &= 94 & \text{(Convertible)} \\
w_5 &= 80 & \text{(Minivan)} \\
w_6 &= 82 & \text{(Coupe)} \\
w_7 &= 71 & \text{(Hatchback)} \\
w_8 &= 100 & \text{(Station Wagon)} \\
w_9 &= 28 & \text{(Electric Car)} \\
w_{10} &= 93 & \text{(Hybrid Car)} \\
w_{11} &= 84 & \text{(Luxury Sedan)} \\
w_{12} &= 83 & \text{(Sports Car)} \\
w_{13} &= 62 & \text{(Crossover)} \\
w_{14} &= 90 & \text{(Diesel Truck)} \\
w_{15} &= 99 & \text{(Compact SUV)} \\
w_{16} &= 96 & \text{(Luxury SUV)} \\
w_{17} &= 21 & \text{(Cargo Van)} \\
w_{18} &= 39 & \text{(Pickup Truck)} \\
w_{19} &= 99 & \text{(Roadster)} \\
w_{20} &= 81 & \text{(Muscle Car)} \\
w_{21} &= 6 & \text{(Off-road Vehicle)} \\
w_{22} &= 58 & \text{(Camper Van)} \\
w_{23} &= 37 & \text{(Compact Car)} \\
w_{24} &= 15 & \text{(Motorcycle)} \\
w_{25} &= 37 & \text{(Electric SUV)} \\
\end{align*}
\]

**Non-negativity and Integrality:**

\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 25
\]

##### Complete Model

\[
\begin{align*}
\max \quad & 2524x_1 + 4614x_2 + 8416x_3 + 5917x_4 + 9048x_5 + 1140x_6 + 8962x_7 + 1888x_8 + 8487x_9 + 4425x_{10} \\
& + 4717x_{11} + 4210x_{12} + 1226x_{13} + 7400x_{14} + 4639x_{15} + 7712x_{16} + 3299x_{17} + 9895x_{18} + 4496x_{19} \\
& + 4526x_{20} + 5688x_{21} + 3007x_{22} + 3623x_{23} + 8474x_{24} + 8372x_{25} \\
\text{s.t.} \quad & 99x_1 + 55x_2 + 75x_3 + 94x_4 + 80x_5 + 82x_6 + 71x_7 + 100x_8 + 28x_9 + 93x_{10} \\
& + 84x_{11} + 83x_{12} + 62x_{13} + 90x_{14} + 99x_{15} + 96x_{16} + 21x_{17} + 39x_{18} + 99x_{19} \\
& + 81x_{20} + 6x_{21} + 58x_{22} + 37x_{23} + 15x_{24} + 37x_{25} \leq 765 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad i = 1, \ldots, 25
\end{align*}
\]