##### Decision Variables

Let $x_i \in \mathbb{Z}_{\geq 0}$ denote the integer scale of development in area $i$ per day, for each area $i$.

##### Parameters

Let $A$ be the set of areas:
- Queens
- Brooklyn
- Manhattan
- Bronx
- Staten Island
- Harlem
- Upper East Side
- Lower Manhattan
- Midtown
- Long Island City
- Williamsburg
- Bushwick
- Flatbush
- Greenpoint
- Park Slope
- Astoria
- Jackson Heights
- Flushing
- Sunnyside
- Ditmars

Let $v_i$ be the benefit coefficient for area $i$:

| Area               | $v_i$ (Value) |
|--------------------|:------------:|
| Queens             | 443          |
| Brooklyn           | 522          |
| Manhattan          | 300          |
| Bronx              | 767          |
| Staten Island      | 300          |
| Harlem             | 309          |
| Upper East Side    | 598          |
| Lower Manhattan    | 460          |
| Midtown            | 318          |
| Long Island City   | 126          |
| Williamsburg       | 593          |
| Bushwick           | 871          |
| Flatbush           | 858          |
| Greenpoint         | 321          |
| Park Slope         | 275          |
| Astoria            | 700          |
| Jackson Heights    | 685          |
| Flushing           | 940          |
| Sunnyside          | 522          |
| Ditmars            | 763          |

Let $w_i$ be the development "weight" (resource usage per unit) for area $i$:

| Area               | $w_i$ (Weight) |
|--------------------|:--------------:|
| Queens             | 104            |
| Brooklyn           | 368            |
| Manhattan          | 483            |
| Bronx              | 165            |
| Staten Island      | 105            |
| Harlem             | 123            |
| Upper East Side    | 131            |
| Lower Manhattan    | 341            |
| Midtown            | 258            |
| Long Island City   | 469            |
| Williamsburg       | 387            |
| Bushwick           | 425            |
| Flatbush           | 482            |
| Greenpoint         | 495            |
| Park Slope         | 305            |
| Astoria            | 377            |
| Jackson Heights    | 318            |
| Flushing           | 56             |
| Sunnyside          | 213            |
| Ditmars            | 472            |

Total available development capacity: $C = 4466$

##### Objective Function

\[
\max \sum_{i \in A} v_i x_i
\]

##### Constraints

1. **Total capacity constraint:**
   \[
   \sum_{i \in A} w_i x_i \leq 4466
   \]

2. **Integrality and nonnegativity:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in A
   \]

##### Complete Model

\[
\begin{align*}
\max \quad & \sum_{i \in A} v_i x_i \\
\text{s.t.} \quad & \sum_{i \in A} w_i x_i \leq 4466 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in A
\end{align*}
\]

###### Retrieved Information

- Areas $A$:
  - Queens, Brooklyn, Manhattan, Bronx, Staten Island, Harlem, Upper East Side, Lower Manhattan, Midtown, Long Island City, Williamsburg, Bushwick, Flatbush, Greenpoint, Park Slope, Astoria, Jackson Heights, Flushing, Sunnyside, Ditmars
- Benefit coefficients $v_i$:
  - Queens: 443, Brooklyn: 522, Manhattan: 300, Bronx: 767, Staten Island: 300, Harlem: 309, Upper East Side: 598, Lower Manhattan: 460, Midtown: 318, Long Island City: 126, Williamsburg: 593, Bushwick: 871, Flatbush: 858, Greenpoint: 321, Park Slope: 275, Astoria: 700, Jackson Heights: 685, Flushing: 940, Sunnyside: 522, Ditmars: 763
- Weights $w_i$:
  - Queens: 104, Brooklyn: 368, Manhattan: 483, Bronx: 165, Staten Island: 105, Harlem: 123, Upper East Side: 131, Lower Manhattan: 341, Midtown: 258, Long Island City: 469, Williamsburg: 387, Bushwick: 425, Flatbush: 482, Greenpoint: 495, Park Slope: 305, Astoria: 377, Jackson Heights: 318, Flushing: 56, Sunnyside: 213, Ditmars: 472
- Total capacity $C = 4466$