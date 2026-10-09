Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $i$ indexes the following ProductName values in the order given below.

**Objective:**
\[
\max \Big(
1752\, x_{\text{Sedan}}
+ 1856\, x_{\text{SUV}}
+ 8372\, x_{\text{Truck}}
+ 6168\, x_{\text{Convertible}}
+ 9681\, x_{\text{Minivan}}
+ 8062\, x_{\text{Coupe}}
+ 3895\, x_{\text{Hatchback}}
+ 3254\, x_{\text{Station Wagon}}
+ 1701\, x_{\text{Electric Car}}
+ 2724\, x_{\text{Luxury Sedan}}
+ 6304\, x_{\text{Sports Car}}
+ 3255\, x_{\text{Crossover}}
+ 1923\, x_{\text{Diesel Truck}}
+ 4103\, x_{\text{Compact SUV}}
+ 4429\, x_{\text{Luxury SUV}}
+ 2663\, x_{\text{Cargo Van}}
+ 1691\, x_{\text{Pickup Truck}}
+ 5632\, x_{\text{Roadster}}
+ 4793\, x_{\text{Muscle Car}}
+ 1343\, x_{\text{Off-road Vehicle}}
+ 9124\, x_{\text{Camper Van}}
+ 3652\, x_{\text{Compact Car}}
+ 8842\, x_{\text{Motorcycle}}
+ 9176\, x_{\text{Electric SUV}}
\Big)
\]

**Subject to:**

\[
15\, x_{\text{Sedan}}
+ 87\, x_{\text{SUV}}
+ 36\, x_{\text{Truck}}
+ 30\, x_{\text{Convertible}}
+ 33\, x_{\text{Minivan}}
+ 72\, x_{\text{Coupe}}
+ 75\, x_{\text{Hatchback}}
+ 71\, x_{\text{Station Wagon}}
+ 51\, x_{\text{Electric Car}}
+ 97\, x_{\text{Luxury Sedan}}
+ 52\, x_{\text{Sports Car}}
+ 25\, x_{\text{Crossover}}
+ 15\, x_{\text{Diesel Truck}}
+ 54\, x_{\text{Compact SUV}}
+ 57\, x_{\text{Luxury SUV}}
+ 18\, x_{\text{Cargo Van}}
+ 69\, x_{\text{Pickup Truck}}
+ 26\, x_{\text{Roadster}}
+ 38\, x_{\text{Muscle Car}}
+ 31\, x_{\text{Off-road Vehicle}}
+ 74\, x_{\text{Camper Van}}
+ 82\, x_{\text{Compact Car}}
+ 49\, x_{\text{Motorcycle}}
+ 64\, x_{\text{Electric SUV}}
\leq 1576
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

**Where:**

- $x_i$ is the number of units of vehicle type $i$ to order daily (integer, $\geq 0$).
- The coefficients in the objective are the "Value" for each ProductName from "products.csv".
- The coefficients in the constraint are the "Weight" for each ProductName from "products.csv".
- The right-hand side of the constraint is the "Capacity" from "capacity.csv".

**ProductName order (as in data):**
1. Sedan
2. SUV
3. Truck
4. Convertible
5. Minivan
6. Coupe
7. Hatchback
8. Station Wagon
9. Electric Car
10. Luxury Sedan
11. Sports Car
12. Crossover
13. Diesel Truck
14. Compact SUV
15. Luxury SUV
16. Cargo Van
17. Pickup Truck
18. Roadster
19. Muscle Car
20. Off-road Vehicle
21. Camper Van
22. Compact Car
23. Motorcycle
24. Electric SUV