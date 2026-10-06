##### Sets and Indices

Let $I$ be the set of areas (from products.csv), indexed by $i$.

##### Parameters

For each area $i$:
- $v_i$ = Value (development benefit) of area $i$
- $w_i$ = Weight (resource requirement) of area $i$

Let $C$ = overall development capacity (from capacity.csv)

##### Decision Variables

For each area $i$:
- $x_i$ = scale of development per day in area $i$ (nonnegative integer)

##### Mathematical Model

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

##### Data

- Capacity (from capacity.csv):

| archive_revision_number | Capacity |
|------------------------|----------|
| 7                      | 586      |

- Areas and their parameters (from products.csv):

| record_keeper_group | archive_revision_number | ProductName         | Value | Weight |
|--------------------|------------------------|---------------------|-------|--------|
| Team B             | 9                      | Queens              | 469   | 954    |
| Team C             | 6                      | Brooklyn            | 290   | 650    |
| Team C             | 1                      | Manhattan           | 236   | 961    |
| Team B             | 2                      | Bronx               | 235   | 950    |
| Team B             | 6                      | Staten Island       | 745   | 379    |
| Team B             | 9                      | Harlem              | 684   | 776    |
| Team A             | 2                      | Upper East Side     | 444   | 381    |
| Team C             | 5                      | Lower Manhattan     | 172   | 808    |
| Team B             | 9                      | Midtown             | 1000  | 937    |
| Team B             | 1                      | Long Island City    | 336   | 608    |
| Team B             | 3                      | Williamsburg        | 546   | 912    |
| Team C             | 1                      | Bushwick            | 535   | 391    |
| Team A             | 5                      | Flatbush            | 539   | 465    |
| Team C             | 7                      | Greenpoint          | 831   | 490    |
| Team A             | 6                      | Park Slope          | 139   | 918    |
| Team B             | 1                      | Astoria             | 432   | 787    |
| Team C             | 1                      | Jackson Heights     | 627   | 347    |
| Team C             | 7                      | Flushing            | 629   | 274    |
| Team A             | 6                      | Sunnyside           | 292   | 642    |
| Team B             | 7                      | Ditmars             | 978   | 130    |

##### Complete Formulation

Let $I$ = {Queens, Brooklyn, Manhattan, Bronx, Staten Island, Harlem, Upper East Side, Lower Manhattan, Midtown, Long Island City, Williamsburg, Bushwick, Flatbush, Greenpoint, Park Slope, Astoria, Jackson Heights, Flushing, Sunnyside, Ditmars}

Let $v_i$ and $w_i$ be as in the table above, and $C = 586$.

$$
\max \Big(
469x_{\text{Queens}} + 290x_{\text{Brooklyn}} + 236x_{\text{Manhattan}} + 235x_{\text{Bronx}} + 745x_{\text{Staten Island}} + 684x_{\text{Harlem}} + 444x_{\text{Upper East Side}} + 172x_{\text{Lower Manhattan}} + 1000x_{\text{Midtown}} + 336x_{\text{Long Island City}} + 546x_{\text{Williamsburg}} + 535x_{\text{Bushwick}} + 539x_{\text{Flatbush}} + 831x_{\text{Greenpoint}} + 139x_{\text{Park Slope}} + 432x_{\text{Astoria}} + 627x_{\text{Jackson Heights}} + 629x_{\text{Flushing}} + 292x_{\text{Sunnyside}} + 978x_{\text{Ditmars}}
\Big)
$$

Subject to:
$$
954x_{\text{Queens}} + 650x_{\text{Brooklyn}} + 961x_{\text{Manhattan}} + 950x_{\text{Bronx}} + 379x_{\text{Staten Island}} + 776x_{\text{Harlem}} + 381x_{\text{Upper East Side}} + 808x_{\text{Lower Manhattan}} + 937x_{\text{Midtown}} + 608x_{\text{Long Island City}} + 912x_{\text{Williamsburg}} + 391x_{\text{Bushwick}} + 465x_{\text{Flatbush}} + 490x_{\text{Greenpoint}} + 918x_{\text{Park Slope}} + 787x_{\text{Astoria}} + 347x_{\text{Jackson Heights}} + 274x_{\text{Flushing}} + 642x_{\text{Sunnyside}} + 130x_{\text{Ditmars}} \leq 586
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$