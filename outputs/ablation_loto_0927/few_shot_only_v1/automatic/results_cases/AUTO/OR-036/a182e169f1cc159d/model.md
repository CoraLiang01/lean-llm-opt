**Mathematical Optimization Model**

Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $i$ indexes the 25 vehicle types listed in the table below. All $x_i$ are required to be nonnegative integers.

**Parameters:**

- $v_i$: Value (benefit) of ordering one unit of vehicle type $i$
- $w_i$: Weight (inventory space required) for one unit of vehicle type $i$
- $C$: Total inventory capacity ($C = 1576$)

**Vehicle Types, Values, and Weights:**

| $i$ | ProductName           | $v_i$ | $w_i$ |
|-----|----------------------|-------|-------|
| 1   | Sedan                | 1752  | 15    |
| 2   | SUV                  | 1856  | 87    |
| 3   | Truck                | 8372  | 36    |
| 4   | Convertible          | 6168  | 30    |
| 5   | Minivan              | 9681  | 33    |
| 6   | Coupe                | 8062  | 72    |
| 7   | Hatchback            | 3895  | 75    |
| 8   | Station Wagon        | 3254  | 71    |
| 9   | Electric Car         | 1701  | 51    |
| 10  | Hybrid Car           | 6799  | 21    |
| 11  | Luxury Sedan         | 2724  | 97    |
| 12  | Sports Car           | 6304  | 52    |
| 13  | Crossover            | 3255  | 25    |
| 14  | Diesel Truck         | 1923  | 15    |
| 15  | Compact SUV          | 4103  | 54    |
| 16  | Luxury SUV           | 4429  | 57    |
| 17  | Cargo Van            | 2663  | 18    |
| 18  | Pickup Truck         | 1691  | 69    |
| 19  | Roadster             | 5632  | 26    |
| 20  | Muscle Car           | 4793  | 38    |
| 21  | Off-road Vehicle     | 1343  | 31    |
| 22  | Camper Van           | 9124  | 74    |
| 23  | Compact Car          | 3652  | 82    |
| 24  | Motorcycle           | 8842  | 49    |
| 25  | Electric SUV         | 9176  | 64    |

---

### Objective

$$
\max \sum_{i=1}^{25} v_i x_i
$$

That is,

\[
\max \Big(
1752\,x_1 + 1856\,x_2 + 8372\,x_3 + 6168\,x_4 + 9681\,x_5 + 8062\,x_6 + 3895\,x_7 + 3254\,x_8 + 1701\,x_9 + 6799\,x_{10} + 2724\,x_{11} + 6304\,x_{12} + 3255\,x_{13} + 1923\,x_{14} + 4103\,x_{15} + 4429\,x_{16} + 2663\,x_{17} + 1691\,x_{18} + 5632\,x_{19} + 4793\,x_{20} + 1343\,x_{21} + 9124\,x_{22} + 3652\,x_{23} + 8842\,x_{24} + 9176\,x_{25}
\Big)
\]

---

### Subject to

**Inventory Capacity Constraint:**

\[
15\,x_1 + 87\,x_2 + 36\,x_3 + 30\,x_4 + 33\,x_5 + 72\,x_6 + 75\,x_7 + 71\,x_8 + 51\,x_9 + 21\,x_{10} + 97\,x_{11} + 52\,x_{12} + 25\,x_{13} + 15\,x_{14} + 54\,x_{15} + 57\,x_{16} + 18\,x_{17} + 69\,x_{18} + 26\,x_{19} + 38\,x_{20} + 31\,x_{21} + 74\,x_{22} + 82\,x_{23} + 49\,x_{24} + 64\,x_{25} \leq 1576
\]

**Integrality and Nonnegativity:**

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
\]

---

**Decision variables:**  
$x_i$ = number of units of vehicle type $i$ to order daily (integer, $\geq 0$)