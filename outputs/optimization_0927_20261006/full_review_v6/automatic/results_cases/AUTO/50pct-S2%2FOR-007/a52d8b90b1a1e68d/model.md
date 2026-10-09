Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following vehicle types (in the order and with the identifiers as given in the data):

\[
\begin{array}{ll}
\text{1. Sedan} & \text{(Value: 2524, Weight: 99)} \\
\text{2. SUV} & \text{(Value: 4614, Weight: 55)} \\
\text{3. Truck} & \text{(Value: 8416, Weight: 75)} \\
\text{4. Convertible} & \text{(Value: 5917, Weight: 94)} \\
\text{5. Minivan} & \text{(Value: 9048, Weight: 80)} \\
\text{6. Coupe} & \text{(Value: 1140, Weight: 82)} \\
\text{7. Hatchback} & \text{(Value: 8962, Weight: 71)} \\
\text{8. Station Wagon} & \text{(Value: 1888, Weight: 100)} \\
\text{9. Electric Car} & \text{(Value: 8487, Weight: 28)} \\
\text{10. Hybrid Car} & \text{(Value: 4425, Weight: 93)} \\
\text{11. Luxury Sedan} & \text{(Value: 4717, Weight: 84)} \\
\text{12. Sports Car} & \text{(Value: 4210, Weight: 83)} \\
\text{13. Crossover} & \text{(Value: 1226, Weight: 62)} \\
\text{14. Diesel Truck} & \text{(Value: 7400, Weight: 90)} \\
\text{15. Compact SUV} & \text{(Value: 4639, Weight: 99)} \\
\text{16. Luxury SUV} & \text{(Value: 7712, Weight: 96)} \\
\text{17. Cargo Van} & \text{(Value: 3299, Weight: 21)} \\
\text{18. Pickup Truck} & \text{(Value: 9895, Weight: 39)} \\
\text{19. Roadster} & \text{(Value: 4496, Weight: 99)} \\
\text{20. Muscle Car} & \text{(Value: 4526, Weight: 81)} \\
\text{21. Off-road Vehicle} & \text{(Value: 5688, Weight: 6)} \\
\text{22. Camper Van} & \text{(Value: 3007, Weight: 58)} \\
\text{23. Compact Car} & \text{(Value: 3623, Weight: 37)} \\
\text{24. Motorcycle} & \text{(Value: 8474, Weight: 15)} \\
\text{25. Electric SUV} & \text{(Value: 8372, Weight: 37)} \\
\end{array}
\]

The overall inventory capacity is $765$ units of weight.

**Mathematical Model:**

**Decision Variables:**
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for each vehicle type } i=1,\ldots,25
\]

**Objective:**
\[
\max \Big(
2524\,x_1 + 4614\,x_2 + 8416\,x_3 + 5917\,x_4 + 9048\,x_5 + 1140\,x_6 + 8962\,x_7 + 1888\,x_8 + 8487\,x_9 + 4425\,x_{10} + 4717\,x_{11} + 4210\,x_{12} + 1226\,x_{13} + 7400\,x_{14} + 4639\,x_{15} + 7712\,x_{16} + 3299\,x_{17} + 9895\,x_{18} + 4496\,x_{19} + 4526\,x_{20} + 5688\,x_{21} + 3007\,x_{22} + 3623\,x_{23} + 8474\,x_{24} + 8372\,x_{25}
\Big)
\]

**Subject to:**

\[
99\,x_1 + 55\,x_2 + 75\,x_3 + 94\,x_4 + 80\,x_5 + 82\,x_6 + 71\,x_7 + 100\,x_8 + 28\,x_9 + 93\,x_{10} + 84\,x_{11} + 83\,x_{12} + 62\,x_{13} + 90\,x_{14} + 99\,x_{15} + 96\,x_{16} + 21\,x_{17} + 39\,x_{18} + 99\,x_{19} + 81\,x_{20} + 6\,x_{21} + 58\,x_{22} + 37\,x_{23} + 15\,x_{24} + 37\,x_{25} \leq 765
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,25
\]

**Where:**

- $x_i$ = number of vehicles of type $i$ to order per day (nonnegative integer)
- The coefficients in the objective are the "Value" (profit) per vehicle from products.csv.
- The coefficients in the constraint are the "Weight" per vehicle from products.csv.
- The right-hand side of the constraint is the "Capacity" from capacity.csv.

**All data and identifiers are preserved in original order as required.**