##### Sets and Indices

Let $I$ be the set of areas:
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

Let $x_i$ be the integer number of development units in area $i$ per day.

##### Parameters

For each area $i$:

| Area                | Value ($v_i$) | Weight ($w_i$) |
|---------------------|--------------|---------------|
| Queens              | 443          | 104           |
| Brooklyn            | 522          | 368           |
| Manhattan           | 300          | 483           |
| Bronx               | 767          | 165           |
| Staten Island       | 300          | 105           |
| Harlem              | 309          | 123           |
| Upper East Side     | 598          | 131           |
| Lower Manhattan     | 460          | 341           |
| Midtown             | 318          | 258           |
| Long Island City    | 126          | 469           |
| Williamsburg        | 593          | 387           |
| Bushwick            | 871          | 425           |
| Flatbush            | 858          | 482           |
| Greenpoint          | 321          | 495           |
| Park Slope          | 275          | 305           |
| Astoria             | 700          | 377           |
| Jackson Heights     | 685          | 318           |
| Flushing            | 940          | 56            |
| Sunnyside           | 522          | 213           |
| Ditmars             | 763          | 472           |

Total development capacity: $C = 4466$

##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$, for all areas $i$.

##### Mathematical Model

Objective:
$$
\max \left(
443\,x_{\text{Queens}} +
522\,x_{\text{Brooklyn}} +
300\,x_{\text{Manhattan}} +
767\,x_{\text{Bronx}} +
300\,x_{\text{Staten Island}} +
309\,x_{\text{Harlem}} +
598\,x_{\text{Upper East Side}} +
460\,x_{\text{Lower Manhattan}} +
318\,x_{\text{Midtown}} +
126\,x_{\text{Long Island City}} +
593\,x_{\text{Williamsburg}} +
871\,x_{\text{Bushwick}} +
858\,x_{\text{Flatbush}} +
321\,x_{\text{Greenpoint}} +
275\,x_{\text{Park Slope}} +
700\,x_{\text{Astoria}} +
685\,x_{\text{Jackson Heights}} +
940\,x_{\text{Flushing}} +
522\,x_{\text{Sunnyside}} +
763\,x_{\text{Ditmars}}
\right)
$$

Subject to:
$$
104\,x_{\text{Queens}} +
368\,x_{\text{Brooklyn}} +
483\,x_{\text{Manhattan}} +
165\,x_{\text{Bronx}} +
105\,x_{\text{Staten Island}} +
123\,x_{\text{Harlem}} +
131\,x_{\text{Upper East Side}} +
341\,x_{\text{Lower Manhattan}} +
258\,x_{\text{Midtown}} +
469\,x_{\text{Long Island City}} +
387\,x_{\text{Williamsburg}} +
425\,x_{\text{Bushwick}} +
482\,x_{\text{Flatbush}} +
495\,x_{\text{Greenpoint}} +
305\,x_{\text{Park Slope}} +
377\,x_{\text{Astoria}} +
318\,x_{\text{Jackson Heights}} +
56\,x_{\text{Flushing}} +
213\,x_{\text{Sunnyside}} +
472\,x_{\text{Ditmars}}
\leq 4466
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$