Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following vehicle types:

\[
\begin{array}{ll}
\text{1. Sedan} & \text{8. Hatchback} \\
\text{2. SUV} & \text{9. Station Wagon} \\
\text{3. Truck} & \text{10. Electric Car} \\
\text{4. Convertible} & \text{11. Hybrid Car} \\
\text{5. Minivan} & \text{12. Luxury Sedan} \\
\text{6. Coupe} & \text{13. Sports Car} \\
\text{7. Crossover} & \text{14. Diesel Truck} \\
\text{15. Compact SUV} & \text{22. Camper Van} \\
\text{16. Luxury SUV} & \text{23. Compact Car} \\
\text{17. Cargo Van} & \text{24. Motorcycle} \\
\text{18. Pickup Truck} & \text{25. Electric SUV} \\
\text{19. Roadster} & \\
\text{20. Muscle Car} & \\
\text{21. Off-road Vehicle} & \\
\end{array}
\]

The profit per unit and weight per unit for each vehicle type are as follows:

\[
\begin{array}{lll}
\text{ProductName} & \text{Value} & \text{Weight} \\
\hline
\text{Sedan} & 2524 & 99 \\
\text{SUV} & 4614 & 55 \\
\text{Truck} & 8416 & 75 \\
\text{Convertible} & 5917 & 94 \\
\text{Minivan} & 9048 & 80 \\
\text{Coupe} & 1140 & 82 \\
\text{Hatchback} & 8962 & 71 \\
\text{Station Wagon} & 1888 & 100 \\
\text{Electric Car} & 8487 & 28 \\
\text{Hybrid Car} & 4425 & 93 \\
\text{Luxury Sedan} & 4717 & 84 \\
\text{Sports Car} & 4210 & 83 \\
\text{Crossover} & 1226 & 62 \\
\text{Diesel Truck} & 7400 & 90 \\
\text{Compact SUV} & 4639 & 99 \\
\text{Luxury SUV} & 7712 & 96 \\
\text{Cargo Van} & 3299 & 21 \\
\text{Pickup Truck} & 9895 & 39 \\
\text{Roadster} & 4496 & 99 \\
\text{Muscle Car} & 4526 & 81 \\
\text{Off-road Vehicle} & 5688 & 6 \\
\text{Camper Van} & 3007 & 58 \\
\text{Compact Car} & 3623 & 37 \\
\text{Motorcycle} & 8474 & 15 \\
\text{Electric SUV} & 8372 & 37 \\
\end{array}
\]

The total inventory capacity is:

\[
765
\]

The mathematical model is:

**Decision Variables:**

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{all vehicle types listed above}\}
\]

**Objective:**

\[
\max \left(
2524\,x_{\text{Sedan}} +
4614\,x_{\text{SUV}} +
8416\,x_{\text{Truck}} +
5917\,x_{\text{Convertible}} +
9048\,x_{\text{Minivan}} +
1140\,x_{\text{Coupe}} +
8962\,x_{\text{Hatchback}} +
1888\,x_{\text{Station Wagon}} +
8487\,x_{\text{Electric Car}} +
4425\,x_{\text{Hybrid Car}} +
4717\,x_{\text{Luxury Sedan}} +
4210\,x_{\text{Sports Car}} +
1226\,x_{\text{Crossover}} +
7400\,x_{\text{Diesel Truck}} +
4639\,x_{\text{Compact SUV}} +
7712\,x_{\text{Luxury SUV}} +
3299\,x_{\text{Cargo Van}} +
9895\,x_{\text{Pickup Truck}} +
4496\,x_{\text{Roadster}} +
4526\,x_{\text{Muscle Car}} +
5688\,x_{\text{Off-road Vehicle}} +
3007\,x_{\text{Camper Van}} +
3623\,x_{\text{Compact Car}} +
8474\,x_{\text{Motorcycle}} +
8372\,x_{\text{Electric SUV}}
\right)
\]

**Subject to:**

\[
99\,x_{\text{Sedan}} +
55\,x_{\text{SUV}} +
75\,x_{\text{Truck}} +
94\,x_{\text{Convertible}} +
80\,x_{\text{Minivan}} +
82\,x_{\text{Coupe}} +
71\,x_{\text{Hatchback}} +
100\,x_{\text{Station Wagon}} +
28\,x_{\text{Electric Car}} +
93\,x_{\text{Hybrid Car}} +
84\,x_{\text{Luxury Sedan}} +
83\,x_{\text{Sports Car}} +
62\,x_{\text{Crossover}} +
90\,x_{\text{Diesel Truck}} +
99\,x_{\text{Compact SUV}} +
96\,x_{\text{Luxury SUV}} +
21\,x_{\text{Cargo Van}} +
39\,x_{\text{Pickup Truck}} +
99\,x_{\text{Roadster}} +
81\,x_{\text{Muscle Car}} +
6\,x_{\text{Off-road Vehicle}} +
58\,x_{\text{Camper Van}} +
37\,x_{\text{Compact Car}} +
15\,x_{\text{Motorcycle}} +
37\,x_{\text{Electric SUV}}
\leq 765
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

**Where:**

- $x_i$ = number of vehicles of type $i$ to order per day (nonnegative integer)
- The coefficients in the objective are the profit per unit for each vehicle type.
- The coefficients in the constraint are the weight (inventory space) per unit for each vehicle type.
- The right-hand side of the constraint is the total inventory capacity.

**All data and identifiers are preserved in original order as retrieved.**