Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following vehicle types in the order given:

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

Define the following parameters (from the data):

- Profits $p_i$:
  - $p_1 = 2524$
  - $p_2 = 4614$
  - $p_3 = 8416$
  - $p_4 = 5917$
  - $p_5 = 9048$
  - $p_6 = 1140$
  - $p_7 = 8962$
  - $p_8 = 1888$
  - $p_9 = 8487$
  - $p_{10} = 4425$
  - $p_{11} = 4717$
  - $p_{12} = 4210$
  - $p_{13} = 1226$
  - $p_{14} = 7400$
  - $p_{15} = 4639$
  - $p_{16} = 7712$
  - $p_{17} = 3299$
  - $p_{18} = 9895$
  - $p_{19} = 4496$
  - $p_{20} = 4526$
  - $p_{21} = 5688$
  - $p_{22} = 3007$
  - $p_{23} = 3623$
  - $p_{24} = 8474$
  - $p_{25} = 8372$

- Weights $w_i$:
  - $w_1 = 99$
  - $w_2 = 55$
  - $w_3 = 75$
  - $w_4 = 94$
  - $w_5 = 80$
  - $w_6 = 82$
  - $w_7 = 71$
  - $w_8 = 100$
  - $w_9 = 28$
  - $w_{10} = 93$
  - $w_{11} = 84$
  - $w_{12} = 83$
  - $w_{13} = 62$
  - $w_{14} = 90$
  - $w_{15} = 99$
  - $w_{16} = 96$
  - $w_{17} = 21$
  - $w_{18} = 39$
  - $w_{19} = 99$
  - $w_{20} = 81$
  - $w_{21} = 6$
  - $w_{22} = 58$
  - $w_{23} = 37$
  - $w_{24} = 15$
  - $w_{25} = 37$

- Total inventory capacity: $C = 765$

The mathematical model is:

**Objective:**
\[
\max \sum_{i=1}^{25} p_i x_i
\]

**Subject to:**
\[
\sum_{i=1}^{25} w_i x_i \leq 765
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
\]

**Where:**

- $x_i$ = number of vehicles of type $i$ to order per day (nonnegative integer)
- $p_i$ = profit per unit of vehicle type $i$ (see above)
- $w_i$ = weight (inventory space required) per unit of vehicle type $i$ (see above)
- $C = 765$ is the total inventory capacity

**Vehicle types (in order):**
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