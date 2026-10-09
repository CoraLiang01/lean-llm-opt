Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $i$ indexes the following ProductName values in the order given below.

Maximize total benefit:
$$
\max \quad 1752\,x_{\text{Sedan}} + 1856\,x_{\text{SUV}} + 8372\,x_{\text{Truck}} + 6168\,x_{\text{Convertible}} + 9681\,x_{\text{Minivan}} + 8062\,x_{\text{Coupe}} + 3895\,x_{\text{Hatchback}} + 3254\,x_{\text{Station Wagon}} + 1701\,x_{\text{Electric Car}} + 6799\,x_{\text{Hybrid Car}} + 2724\,x_{\text{Luxury Sedan}} + 6304\,x_{\text{Sports Car}} + 3255\,x_{\text{Crossover}} + 1923\,x_{\text{Diesel Truck}} + 4103\,x_{\text{Compact SUV}} + 4429\,x_{\text{Luxury SUV}} + 2663\,x_{\text{Cargo Van}} + 1691\,x_{\text{Pickup Truck}} + 5632\,x_{\text{Roadster}} + 4793\,x_{\text{Muscle Car}} + 1343\,x_{\text{Off-road Vehicle}} + 9124\,x_{\text{Camper Van}} + 3652\,x_{\text{Compact Car}} + 8842\,x_{\text{Motorcycle}} + 9176\,x_{\text{Electric SUV}}
$$

Subject to the total inventory capacity constraint:
$$
15\,x_{\text{Sedan}} + 87\,x_{\text{SUV}} + 36\,x_{\text{Truck}} + 30\,x_{\text{Convertible}} + 33\,x_{\text{Minivan}} + 72\,x_{\text{Coupe}} + 75\,x_{\text{Hatchback}} + 71\,x_{\text{Station Wagon}} + 51\,x_{\text{Electric Car}} + 21\,x_{\text{Hybrid Car}} + 97\,x_{\text{Luxury Sedan}} + 52\,x_{\text{Sports Car}} + 25\,x_{\text{Crossover}} + 15\,x_{\text{Diesel Truck}} + 54\,x_{\text{Compact SUV}} + 57\,x_{\text{Luxury SUV}} + 18\,x_{\text{Cargo Van}} + 69\,x_{\text{Pickup Truck}} + 26\,x_{\text{Roadster}} + 38\,x_{\text{Muscle Car}} + 31\,x_{\text{Off-road Vehicle}} + 74\,x_{\text{Camper Van}} + 82\,x_{\text{Compact Car}} + 49\,x_{\text{Motorcycle}} + 64\,x_{\text{Electric SUV}} \leq 1576
$$

Variable domains:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all vehicle types } i
$$

Where:

- The benefit coefficients are given by the "Value" column for each ProductName.
- The inventory consumption coefficients are given by the "Weight" column for each ProductName.
- The total inventory capacity is 1576 (from the "Capacity" column in capacity.csv).
- All variables are nonnegative integers.

ProductName order (as in products.csv):

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