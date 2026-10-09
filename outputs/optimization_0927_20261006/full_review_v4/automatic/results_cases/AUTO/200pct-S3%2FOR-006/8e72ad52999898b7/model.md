Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $i$ indexes the following ProductName values in the order retrieved.

#### Sets and Parameters

- Let $I$ be the set of vehicle types (ProductName):

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

- For each $i \in I$:
    - $b_i$ = Value (benefit coefficient)
    - $w_i$ = Weight (inventory space requirement per unit)

- Let $C$ = 1576 (total inventory Capacity from "capacity.csv")

#### Decision Variables

- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$ (number of units of vehicle type $i$ to order daily)

#### Objective

$$
\max \sum_{i \in I} b_i x_i
$$

#### Constraints

1. **Total Inventory Capacity:**
   $$
   \sum_{i \in I} w_i x_i \leq 1576
   $$

2. **Integrality and Nonnegativity:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

#### Data Table (in source order):

| ProductName        | Value | Weight |
|--------------------|-------|--------|
| Sedan              | 1752  | 15     |
| SUV                | 1856  | 87     |
| Truck              | 8372  | 36     |
| Convertible        | 6168  | 30     |
| Minivan            | 9681  | 33     |
| Coupe              | 8062  | 72     |
| Hatchback          | 3895  | 75     |
| Station Wagon      | 3254  | 71     |
| Electric Car       | 1701  | 51     |
| Hybrid Car         | 6799  | 21     |
| Luxury Sedan       | 2724  | 97     |
| Sports Car         | 6304  | 52     |
| Crossover          | 3255  | 25     |
| Diesel Truck       | 1923  | 15     |
| Compact SUV        | 4103  | 54     |
| Luxury SUV         | 4429  | 57     |
| Cargo Van          | 2663  | 18     |
| Pickup Truck       | 1691  | 69     |
| Roadster           | 5632  | 26     |
| Muscle Car         | 4793  | 38     |
| Off-road Vehicle   | 1343  | 31     |
| Camper Van         | 9124  | 74     |
| Compact Car        | 3652  | 82     |
| Motorcycle         | 8842  | 49     |
| Electric SUV       | 9176  | 64     |

#### Complete Model

$$
\begin{align*}
\max\quad & 1752x_1 + 1856x_2 + 8372x_3 + 6168x_4 + 9681x_5 + 8062x_6 + 3895x_7 + 3254x_8 \\
& + 1701x_9 + 6799x_{10} + 2724x_{11} + 6304x_{12} + 3255x_{13} + 1923x_{14} + 4103x_{15} \\
& + 4429x_{16} + 2663x_{17} + 1691x_{18} + 5632x_{19} + 4793x_{20} + 1343x_{21} + 9124x_{22} \\
& + 3652x_{23} + 8842x_{24} + 9176x_{25} \\
\text{s.t.}\quad & 15x_1 + 87x_2 + 36x_3 + 30x_4 + 33x_5 + 72x_6 + 75x_7 + 71x_8 \\
& + 51x_9 + 21x_{10} + 97x_{11} + 52x_{12} + 25x_{13} + 15x_{14} + 54x_{15} \\
& + 57x_{16} + 18x_{17} + 69x_{18} + 26x_{19} + 38x_{20} + 31x_{21} + 74x_{22} \\
& + 82x_{23} + 49x_{24} + 64x_{25} \leq 1576 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,25
\end{align*}
$$

Where $x_1$ = Sedan, $x_2$ = SUV, ..., $x_{25}$ = Electric SUV, as per the table above.