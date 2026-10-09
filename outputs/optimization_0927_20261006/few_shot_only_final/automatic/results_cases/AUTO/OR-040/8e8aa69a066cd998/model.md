Let $x_i$ = number of development units in area $i$ (for each ProductName in products.csv), $x_i \in \mathbb{Z}_{\geq 0}$.

**Parameters:**

- For each area $i$ (ProductName):
    - $p_i$ = Value (benefit coefficient)
    - $w_i$ = Weight (resource consumption per unit)
- $C$ = 4466 (overall capacity)

**Model:**

Maximize:
$$
\sum_{i} p_i x_i
$$

Subject to:
$$
\sum_{i} w_i x_i \leq 4466
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

**Where:**

| $i$ (ProductName)      | $p_i$ (Value) | $w_i$ (Weight) |
|------------------------|--------------|---------------|
| Queens                 | 443          | 104           |
| Brooklyn               | 522          | 368           |
| Manhattan              | 300          | 483           |
| Bronx                  | 767          | 165           |
| Staten Island          | 300          | 105           |
| Harlem                 | 309          | 123           |
| Upper East Side        | 598          | 131           |
| Lower Manhattan        | 460          | 341           |
| Midtown                | 318          | 258           |
| Long Island City       | 126          | 469           |
| Williamsburg           | 593          | 387           |
| Bushwick               | 871          | 425           |
| Flatbush               | 858          | 482           |
| Greenpoint             | 321          | 495           |
| Park Slope             | 275          | 305           |
| Astoria                | 700          | 377           |
| Jackson Heights        | 685          | 318           |
| Flushing               | 940          | 56            |
| Sunnyside              | 522          | 213           |
| Ditmars                | 763          | 472           |

**Decision variables:**  
$x_i$ for each area $i$ above, integer and $\geq 0$.