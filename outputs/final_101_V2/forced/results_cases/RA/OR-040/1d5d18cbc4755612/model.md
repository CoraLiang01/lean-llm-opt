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

##### Parameters

Let $v_i$ be the benefit coefficient for area $i$:

- $v_{\text{Queens}} = 443$
- $v_{\text{Brooklyn}} = 522$
- $v_{\text{Manhattan}} = 300$
- $v_{\text{Bronx}} = 767$
- $v_{\text{Staten Island}} = 300$
- $v_{\text{Harlem}} = 309$
- $v_{\text{Upper East Side}} = 598$
- $v_{\text{Lower Manhattan}} = 460$
- $v_{\text{Midtown}} = 318$
- $v_{\text{Long Island City}} = 126$
- $v_{\text{Williamsburg}} = 593$
- $v_{\text{Bushwick}} = 871$
- $v_{\text{Flatbush}} = 858$
- $v_{\text{Greenpoint}} = 321$
- $v_{\text{Park Slope}} = 275$
- $v_{\text{Astoria}} = 700$
- $v_{\text{Jackson Heights}} = 685$
- $v_{\text{Flushing}} = 940$
- $v_{\text{Sunnyside}} = 522$
- $v_{\text{Ditmars}} = 763$

Let $w_i$ be the weight (resource consumption per unit) for area $i$:

- $w_{\text{Queens}} = 104$
- $w_{\text{Brooklyn}} = 368$
- $w_{\text{Manhattan}} = 483$
- $w_{\text{Bronx}} = 165$
- $w_{\text{Staten Island}} = 105$
- $w_{\text{Harlem}} = 123$
- $w_{\text{Upper East Side}} = 131$
- $w_{\text{Lower Manhattan}} = 341$
- $w_{\text{Midtown}} = 258$
- $w_{\text{Long Island City}} = 469$
- $w_{\text{Williamsburg}} = 593$
- $w_{\text{Bushwick}} = 425$
- $w_{\text{Flatbush}} = 482$
- $w_{\text{Greenpoint}} = 495$
- $w_{\text{Park Slope}} = 305$
- $w_{\text{Astoria}} = 700$
- $w_{\text{Jackson Heights}} = 685$
- $w_{\text{Flushing}} = 56$
- $w_{\text{Sunnyside}} = 213$
- $w_{\text{Ditmars}} = 472$

Let $C = 4466$ be the overall development capacity.

##### Decision Variables

For each area $i$, let $x_i$ be the scale of development in area $i$ per day.

$x_i \in \mathbb{Z}_{\geq 0}$ for all $i$.

##### Mathematical Model

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq 4466
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

##### Data (in source order)

| Area               | $v_i$ | $w_i$ |
|--------------------|-------|-------|
| Queens             | 443   | 104   |
| Brooklyn           | 522   | 368   |
| Manhattan          | 300   | 483   |
| Bronx              | 767   | 165   |
| Staten Island      | 300   | 105   |
| Harlem             | 309   | 123   |
| Upper East Side    | 598   | 131   |
| Lower Manhattan    | 460   | 341   |
| Midtown            | 318   | 258   |
| Long Island City   | 126   | 469   |
| Williamsburg       | 593   | 387   |
| Bushwick           | 871   | 425   |
| Flatbush           | 858   | 482   |
| Greenpoint         | 321   | 495   |
| Park Slope         | 275   | 305   |
| Astoria            | 700   | 377   |
| Jackson Heights    | 685   | 318   |
| Flushing           | 940   | 56    |
| Sunnyside          | 522   | 213   |
| Ditmars            | 763   | 472   |