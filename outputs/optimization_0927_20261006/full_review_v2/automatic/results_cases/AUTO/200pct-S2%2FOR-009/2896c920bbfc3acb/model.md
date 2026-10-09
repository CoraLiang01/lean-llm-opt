Let $I$ be the set of areas (indexed by $i$), with area names and parameters as below. Let $x_i$ be the scale of development per day in area $i$ (nonnegative integer).

**Parameters (from products.csv and capacity.csv, in source order):**

| Area (ProductName)      | Value ($v_i$) | Weight ($w_i$) |
|------------------------|--------------|---------------|
| Queens                 | 469          | 954           |
| Brooklyn               | 290          | 650           |
| Manhattan              | 236          | 961           |
| Bronx                  | 235          | 950           |
| Staten Island          | 745          | 379           |
| Harlem                 | 684          | 776           |
| Upper East Side        | 444          | 381           |
| Lower Manhattan        | 172          | 808           |
| Midtown                | 1000         | 937           |
| Long Island City       | 336          | 608           |
| Williamsburg           | 546          | 912           |
| Bushwick               | 535          | 391           |
| Flatbush               | 539          | 465           |
| Greenpoint             | 831          | 490           |
| Park Slope             | 139          | 918           |
| Astoria                | 432          | 787           |
| Jackson Heights        | 627          | 347           |
| Flushing               | 629          | 274           |
| Sunnyside              | 292          | 642           |
| Ditmars                | 978          | 130           |

Overall development capacity (from capacity.csv):

$\text{Capacity} = 586$

---

### Mathematical Model

**Decision variables:**

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

**Objective:**

$$
\max \sum_{i \in I} v_i x_i
$$

**Subject to:**

$$
\sum_{i \in I} w_i x_i \leq 586
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

**Where:**

- $I$ = {Queens, Brooklyn, Manhattan, Bronx, Staten Island, Harlem, Upper East Side, Lower Manhattan, Midtown, Long Island City, Williamsburg, Bushwick, Flatbush, Greenpoint, Park Slope, Astoria, Jackson Heights, Flushing, Sunnyside, Ditmars}
- $v_i$ = Value for area $i$ (see table above)
- $w_i$ = Weight for area $i$ (see table above)

**Complete numerical formulation:**

$$
\max \Big[ 469x_{\text{Queens}} + 290x_{\text{Brooklyn}} + 236x_{\text{Manhattan}} + 235x_{\text{Bronx}} + 745x_{\text{Staten Island}} + 684x_{\text{Harlem}} + 444x_{\text{Upper East Side}} + 172x_{\text{Lower Manhattan}} + 1000x_{\text{Midtown}} + 336x_{\text{Long Island City}} + 546x_{\text{Williamsburg}} + 535x_{\text{Bushwick}} + 539x_{\text{Flatbush}} + 831x_{\text{Greenpoint}} + 139x_{\text{Park Slope}} + 432x_{\text{Astoria}} + 627x_{\text{Jackson Heights}} + 629x_{\text{Flushing}} + 292x_{\text{Sunnyside}} + 978x_{\text{Ditmars}} \Big]
$$

Subject to:

$$
954x_{\text{Queens}} + 650x_{\text{Brooklyn}} + 961x_{\text{Manhattan}} + 950x_{\text{Bronx}} + 379x_{\text{Staten Island}} + 776x_{\text{Harlem}} + 381x_{\text{Upper East Side}} + 808x_{\text{Lower Manhattan}} + 937x_{\text{Midtown}} + 608x_{\text{Long Island City}} + 912x_{\text{Williamsburg}} + 391x_{\text{Bushwick}} + 465x_{\text{Flatbush}} + 490x_{\text{Greenpoint}} + 918x_{\text{Park Slope}} + 787x_{\text{Astoria}} + 347x_{\text{Jackson Heights}} + 274x_{\text{Flushing}} + 642x_{\text{Sunnyside}} + 130x_{\text{Ditmars}} \leq 586
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$