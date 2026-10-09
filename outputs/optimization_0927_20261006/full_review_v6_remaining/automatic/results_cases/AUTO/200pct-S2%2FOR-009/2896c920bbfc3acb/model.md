Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas (ProductName):

1. Queens
2. Brooklyn
3. Manhattan
4. Bronx
5. Staten Island
6. Harlem
7. Upper East Side
8. Lower Manhattan
9. Midtown
10. Long Island City
11. Williamsburg
12. Bushwick
13. Flatbush
14. Greenpoint
15. Park Slope
16. Astoria
17. Jackson Heights
18. Flushing
19. Sunnyside
20. Ditmars

Let $v_i$ be the Value (development benefit) for area $i$ and $w_i$ be the Weight (resource requirement) for area $i$, as given below:

| Area (ProductName)      | $v_i$ (Value) | $w_i$ (Weight) |
|------------------------|:-------------:|:--------------:|
| Queens                 | 469           | 954            |
| Brooklyn               | 290           | 650            |
| Manhattan              | 236           | 961            |
| Bronx                  | 235           | 950            |
| Staten Island          | 745           | 379            |
| Harlem                 | 684           | 776            |
| Upper East Side        | 444           | 381            |
| Lower Manhattan        | 172           | 808            |
| Midtown                | 1000          | 937            |
| Long Island City       | 336           | 608            |
| Williamsburg           | 546           | 912            |
| Bushwick               | 535           | 391            |
| Flatbush               | 539           | 465            |
| Greenpoint             | 831           | 490            |
| Park Slope             | 139           | 918            |
| Astoria                | 432           | 787            |
| Jackson Heights        | 627           | 347            |
| Flushing               | 629           | 274            |
| Sunnyside              | 292           | 642            |
| Ditmars                | 978           | 130            |

The overall development capacity is:

$\text{Capacity} = 586$

##### Mathematical Model

**Decision variables:**
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{areas listed above}\}
$$

**Objective:**
$$
\max \sum_{i=1}^{20} v_i x_i
$$

**Subject to:**
$$
\sum_{i=1}^{20} w_i x_i \leq 586
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

**Where:**

- $v_i$ and $w_i$ are as listed above for each area $i$.
- The capacity constraint uses the value $586$ from the "Capacity" column in capacity.csv.

**All coefficients and identifiers are as retrieved and in source order.**