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

Parameters:

- $p_i$ = Value (profit per unit) for product $i$:
  - Sedan: 2524
  - SUV: 4614
  - Truck: 8416
  - Convertible: 5917
  - Minivan: 9048
  - Coupe: 1140
  - Hatchback: 8962
  - Station Wagon: 1888
  - Electric Car: 8487
  - Hybrid Car: 4425
  - Luxury Sedan: 4717
  - Sports Car: 4210
  - Crossover: 1226
  - Diesel Truck: 7400
  - Compact SUV: 4639
  - Luxury SUV: 7712
  - Cargo Van: 3299
  - Pickup Truck: 9895
  - Roadster: 4496
  - Muscle Car: 4526
  - Off-road Vehicle: 5688
  - Camper Van: 3007
  - Compact Car: 3623
  - Motorcycle: 8474
  - Electric SUV: 8372

- $w_i$ = Weight (inventory space per unit) for product $i$:
  - Sedan: 99
  - SUV: 55
  - Truck: 75
  - Convertible: 94
  - Minivan: 80
  - Coupe: 82
  - Hatchback: 71
  - Station Wagon: 100
  - Electric Car: 28
  - Hybrid Car: 93
  - Luxury Sedan: 84
  - Sports Car: 83
  - Crossover: 62
  - Diesel Truck: 90
  - Compact SUV: 99
  - Luxury SUV: 96
  - Cargo Van: 21
  - Pickup Truck: 39
  - Roadster: 99
  - Muscle Car: 81
  - Off-road Vehicle: 6
  - Camper Van: 58
  - Compact Car: 37
  - Motorcycle: 15
  - Electric SUV: 37

- $C$ = 765 (overall inventory capacity, from Capacity column in capacity.csv)

Objective:
\[
\max \sum_{i=1}^{25} p_i x_i
\]

Subject to:

Inventory capacity constraint:
\[
\sum_{i=1}^{25} w_i x_i \leq 765
\]

Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
\]

Where all $p_i$ and $w_i$ are as listed above, and $x_i$ is the number of vehicles of type $i$ to order per day.