Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following ProductName values in the order returned:

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

Let $p_i$ be the Value for product $i$, and $w_i$ be the Weight for product $i$ (interpreted as the space or capacity requirement per unit). The total available inventory capacity is $C = 765$.

##### Objective:
\[
\max \sum_{i=1}^{25} p_i x_i
\]
where the $p_i$ are as follows (in the order above):

\[
\begin{align*}
p_1 &= 2524 \\
p_2 &= 4614 \\
p_3 &= 8416 \\
p_4 &= 5917 \\
p_5 &= 9048 \\
p_6 &= 1140 \\
p_7 &= 8962 \\
p_8 &= 1888 \\
p_9 &= 8487 \\
p_{10} &= 4425 \\
p_{11} &= 4717 \\
p_{12} &= 4210 \\
p_{13} &= 1226 \\
p_{14} &= 7400 \\
p_{15} &= 4639 \\
p_{16} &= 7712 \\
p_{17} &= 3299 \\
p_{18} &= 9895 \\
p_{19} &= 4496 \\
p_{20} &= 4526 \\
p_{21} &= 5688 \\
p_{22} &= 3007 \\
p_{23} &= 3623 \\
p_{24} &= 8474 \\
p_{25} &= 8372 \\
\end{align*}
\]

##### Subject to:

**Capacity constraint:**
\[
\sum_{i=1}^{25} w_i x_i \leq 765
\]
where the $w_i$ are as follows (in the order above):

\[
\begin{align*}
w_1 &= 99 \\
w_2 &= 55 \\
w_3 &= 75 \\
w_4 &= 94 \\
w_5 &= 80 \\
w_6 &= 82 \\
w_7 &= 71 \\
w_8 &= 100 \\
w_9 &= 28 \\
w_{10} &= 93 \\
w_{11} &= 84 \\
w_{12} &= 83 \\
w_{13} &= 62 \\
w_{14} &= 90 \\
w_{15} &= 99 \\
w_{16} &= 96 \\
w_{17} &= 21 \\
w_{18} &= 39 \\
w_{19} &= 99 \\
w_{20} &= 81 \\
w_{21} &= 6 \\
w_{22} &= 58 \\
w_{23} &= 37 \\
w_{24} &= 15 \\
w_{25} &= 37 \\
\end{align*}
\]

**Nonnegativity and integrality:**
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
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
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
\end{align*}
\]

Where each $x_i$ is the number of vehicles of type $i$ (ProductName as listed above) to order per day.