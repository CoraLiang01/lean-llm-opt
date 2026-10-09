Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following vehicle types:

\[
\begin{array}{ll}
\text{1. Sedan} & \text{Value: } 2524, \quad \text{Weight: } 99 \\
\text{2. SUV} & \text{Value: } 4614, \quad \text{Weight: } 55 \\
\text{3. Truck} & \text{Value: } 8416, \quad \text{Weight: } 75 \\
\text{4. Convertible} & \text{Value: } 5917, \quad \text{Weight: } 94 \\
\text{5. Minivan} & \text{Value: } 9048, \quad \text{Weight: } 80 \\
\text{6. Coupe} & \text{Value: } 1140, \quad \text{Weight: } 82 \\
\text{7. Hatchback} & \text{Value: } 8962, \quad \text{Weight: } 71 \\
\text{8. Station Wagon} & \text{Value: } 1888, \quad \text{Weight: } 100 \\
\text{9. Electric Car} & \text{Value: } 8487, \quad \text{Weight: } 28 \\
\text{10. Hybrid Car} & \text{Value: } 4425, \quad \text{Weight: } 93 \\
\text{11. Luxury Sedan} & \text{Value: } 4717, \quad \text{Weight: } 84 \\
\text{12. Sports Car} & \text{Value: } 4210, \quad \text{Weight: } 83 \\
\text{13. Crossover} & \text{Value: } 1226, \quad \text{Weight: } 62 \\
\text{14. Diesel Truck} & \text{Value: } 7400, \quad \text{Weight: } 90 \\
\text{15. Compact SUV} & \text{Value: } 4639, \quad \text{Weight: } 99 \\
\text{16. Luxury SUV} & \text{Value: } 7712, \quad \text{Weight: } 96 \\
\text{17. Cargo Van} & \text{Value: } 3299, \quad \text{Weight: } 21 \\
\text{18. Pickup Truck} & \text{Value: } 9895, \quad \text{Weight: } 39 \\
\text{19. Roadster} & \text{Value: } 4496, \quad \text{Weight: } 99 \\
\text{20. Muscle Car} & \text{Value: } 4526, \quad \text{Weight: } 81 \\
\text{21. Off-road Vehicle} & \text{Value: } 5688, \quad \text{Weight: } 6 \\
\text{22. Camper Van} & \text{Value: } 3007, \quad \text{Weight: } 58 \\
\text{23. Compact Car} & \text{Value: } 3623, \quad \text{Weight: } 37 \\
\text{24. Motorcycle} & \text{Value: } 8474, \quad \text{Weight: } 15 \\
\text{25. Electric SUV} & \text{Value: } 8372, \quad \text{Weight: } 37 \\
\end{array}
\]

The total inventory capacity is $765$.

---

**Mathematical Model:**

**Decision Variables:**

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{all vehicle types listed above}\}
\]

**Objective:**

\[
\max \Big(
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
\Big)
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

---

**Where:**

- $x_i$ = number of vehicles of type $i$ to order per day (nonnegative integer)
- "Value" = profit per unit for each vehicle type
- "Weight" = inventory space consumed per unit for each vehicle type
- Total inventory space used cannot exceed $765$ units

All coefficients and identifiers are as retrieved and in original order.