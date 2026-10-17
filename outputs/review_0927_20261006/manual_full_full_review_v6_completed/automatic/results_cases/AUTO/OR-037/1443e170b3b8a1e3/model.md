Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following vehicle types:

\[
\begin{array}{ll}
\text{1. Sedan} & \text{8. Hatchback} \\
\text{2. SUV} & \text{9. Station Wagon} \\
\text{3. Truck} & \text{10. Electric Car} \\
\text{4. Convertible} & \text{11. Hybrid Car} \\
\text{5. Minivan} & \text{12. Luxury Sedan} \\
\text{6. Coupe} & \text{13. Sports Car} \\
\text{7. Compact SUV} & \text{14. Crossover} \\
\text{15. Diesel Truck} & \text{22. Camper Van} \\
\text{16. Compact SUV} & \text{23. Compact Car} \\
\text{17. Luxury SUV} & \text{24. Motorcycle} \\
\text{18. Cargo Van} & \text{25. Electric SUV} \\
\text{19. Pickup Truck} & \\
\text{20. Roadster} & \\
\text{21. Muscle Car} & \\
\text{22. Off-road Vehicle} & \\
\end{array}
\]

#### Parameters

- $p_i$: Profit per unit of vehicle $i$ (from "Value" column)
- $w_i$: Space requirement per unit of vehicle $i$ (from "Weight" column)
- $C$: Total inventory capacity (from "Capacity" in capacity.csv)

#### Data

\[
\begin{array}{lll}
\text{ProductName} & p_i & w_i \\
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

Total inventory capacity: $C = 765$

#### Mathematical Model

**Decision variables:**
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

**Objective:**
\[
\max \sum_{i} p_i x_i
\]

**Subject to:**
\[
\sum_{i} w_i x_i \leq 765
\]
\[
x_i \geq 0 \text{ and integer} \quad \forall i
\]

**Where:**

\[
\begin{array}{lll}
x_1 = \text{number of Sedan ordered per day} \\
x_2 = \text{number of SUV ordered per day} \\
x_3 = \text{number of Truck ordered per day} \\
x_4 = \text{number of Convertible ordered per day} \\
x_5 = \text{number of Minivan ordered per day} \\
x_6 = \text{number of Coupe ordered per day} \\
x_7 = \text{number of Hatchback ordered per day} \\
x_8 = \text{number of Station Wagon ordered per day} \\
x_9 = \text{number of Electric Car ordered per day} \\
x_{10} = \text{number of Hybrid Car ordered per day} \\
x_{11} = \text{number of Luxury Sedan ordered per day} \\
x_{12} = \text{number of Sports Car ordered per day} \\
x_{13} = \text{number of Crossover ordered per day} \\
x_{14} = \text{number of Diesel Truck ordered per day} \\
x_{15} = \text{number of Compact SUV ordered per day} \\
x_{16} = \text{number of Luxury SUV ordered per day} \\
x_{17} = \text{number of Cargo Van ordered per day} \\
x_{18} = \text{number of Pickup Truck ordered per day} \\
x_{19} = \text{number of Roadster ordered per day} \\
x_{20} = \text{number of Muscle Car ordered per day} \\
x_{21} = \text{number of Off-road Vehicle ordered per day} \\
x_{22} = \text{number of Camper Van ordered per day} \\
x_{23} = \text{number of Compact Car ordered per day} \\
x_{24} = \text{number of Motorcycle ordered per day} \\
x_{25} = \text{number of Electric SUV ordered per day} \\
\end{array}
\]

**All coefficients and identifiers are as retrieved.**