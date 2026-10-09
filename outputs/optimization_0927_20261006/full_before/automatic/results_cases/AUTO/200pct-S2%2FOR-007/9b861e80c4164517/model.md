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

The profit per unit for each vehicle type $i$ is $p_i$ (from the Value column), and the space required per unit is $w_i$ (from the Weight column). The total inventory capacity is $C = 765$.

The complete mathematical model is:

Objective:
$$
\max \; 2524x_1 + 4614x_2 + 8416x_3 + 5917x_4 + 9048x_5 + 1140x_6 + 8962x_7 + 1888x_8 + 8487x_9 + 4425x_{10} + 4717x_{11} + 4210x_{12} + 1226x_{13} + 7400x_{14} + 4639x_{15} + 7712x_{16} + 3299x_{17} + 9895x_{18} + 4496x_{19} + 4526x_{20} + 5688x_{21} + 3007x_{22} + 3623x_{23} + 8474x_{24} + 8372x_{25}
$$

Subject to:

Capacity constraint:
$$
99x_1 + 55x_2 + 75x_3 + 94x_4 + 80x_5 + 82x_6 + 71x_7 + 100x_8 + 28x_9 + 93x_{10} + 84x_{11} + 83x_{12} + 62x_{13} + 90x_{14} + 99x_{15} + 96x_{16} + 21x_{17} + 39x_{18} + 99x_{19} + 81x_{20} + 6x_{21} + 58x_{22} + 37x_{23} + 15x_{24} + 37x_{25} \leq 765
$$

Integrality and nonnegativity:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
$$

Where:
- $x_i$ = number of vehicles of type $i$ to order per day
- $p_i$ = Value (profit per unit) for vehicle type $i$
- $w_i$ = Weight (space per unit) for vehicle type $i$
- $C = 765$ (total inventory capacity)