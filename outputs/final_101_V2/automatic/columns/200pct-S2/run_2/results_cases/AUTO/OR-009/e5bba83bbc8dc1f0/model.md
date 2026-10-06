##### Sets and Indices

Let $I$ be the set of areas (from ProductName in products.csv):
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

Let $x_i$ = scale of development per day in area $i$, for each $i \in I$.

##### Parameters

For each area $i$:
- $v_i$ = Value (development benefit) of area $i$ (from products.csv)
- $w_i$ = Weight (resource consumption per unit development) of area $i$ (from products.csv)

Total development capacity (from capacity.csv):
- $C = 586$

##### Data

| Area                | $v_i$ (Value) | $w_i$ (Weight) |
|---------------------|--------------|---------------|
| Queens              | 469          | 954           |
| Brooklyn            | 290          | 650           |
| Manhattan           | 236          | 961           |
| Bronx               | 235          | 950           |
| Staten Island       | 745          | 379           |
| Harlem              | 684          | 776           |
| Upper East Side     | 444          | 381           |
| Lower Manhattan     | 172          | 808           |
| Midtown             | 1000         | 937           |
| Long Island City    | 336          | 608           |
| Williamsburg        | 546          | 912           |
| Bushwick            | 535          | 391           |
| Flatbush            | 539          | 465           |
| Greenpoint          | 831          | 490           |
| Park Slope          | 139          | 918           |
| Astoria             | 432          | 787           |
| Jackson Heights     | 627          | 347           |
| Flushing            | 629          | 274           |
| Sunnyside           | 292          | 642           |
| Ditmars             | 978          | 130           |

##### Mathematical Model

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq 586
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

##### Where

- $x_i$ = scale of development per day in area $i$ (nonnegative integer)
- $v_i$ = Value of area $i$ (see table above)
- $w_i$ = Weight of area $i$ (see table above)
- $586$ = total development capacity

##### Retrieved Data

products.csv

| marketing_region_group | nearby_transit_stop_count | marketing_campaign_format | neighborhood_walkability_score | ProductName         | supplier_contact_channel | Value | neighborhood_public_park_count | Weight |
|-----------------------|--------------------------|--------------------------|-------------------------------|---------------------|-------------------------|-------|-------------------------------|--------|
| Campaign East         | 7                        | Brochure                 | 83                            | Queens              | Phone                   | 469   | 4                             | 954    |
| Campaign West         | 3                        | Brochure                 | 42                            | Brooklyn            | Phone                   | 290   | 1                             | 650    |
| Campaign East         | 7                        | Newsletter               | 71                            | Manhattan           | Portal                  | 236   | 5                             | 961    |
| Campaign Central      | 2                        | Brochure                 | 65                            | Bronx               | Email                   | 235   | 1                             | 950    |
| Campaign Central      | 3                        | Brochure                 | 42                            | Staten Island       | Portal                  | 745   | 1                             | 379    |
| Campaign West         | 10                       | Newsletter               | 71                            | Harlem              | Phone                   | 684   | 2                             | 776    |
| Campaign West         | 2                        | Web feature              | 92                            | Upper East Side     | Phone                   | 444   | 2                             | 381    |
| Campaign Central      | 7                        | Newsletter               | 65                            | Lower Manhattan     | Portal                  | 172   | 1                             | 808    |
| Campaign West         | 5                        | Newsletter               | 71                            | Midtown             | Portal                  | 1000  | 2                             | 937    |
| Campaign East         | 7                        | Brochure                 | 92                            | Long Island City    | Email                   | 336   | 1                             | 608    |
| Campaign Central      | 5                        | Brochure                 | 56                            | Williamsburg        | Portal                  | 546   | 1                             | 912    |
| Campaign Central      | 7                        | Brochure                 | 83                            | Bushwick            | Phone                   | 535   | 1                             | 391    |
| Campaign East         | 5                        | Web feature              | 42                            | Flatbush            | Phone                   | 539   | 4                             | 465    |
| Campaign West         | 2                        | Newsletter               | 71                            | Greenpoint          | Email                   | 831   | 3                             | 490    |
| Campaign East         | 10                       | Web feature              | 71                            | Park Slope          | Phone                   | 139   | 1                             | 918    |
| Campaign West         | 2                        | Web feature              | 56                            | Astoria             | Portal                  | 432   | 5                             | 787    |
| Campaign West         | 7                        | Web feature              | 65                            | Jackson Heights     | Portal                  | 627   | 1                             | 347    |
| Campaign Central      | 3                        | Newsletter               | 83                            | Flushing            | Portal                  | 629   | 1                             | 274    |
| Campaign East         | 2                        | Newsletter               | 83                            | Sunnyside           | Email                   | 292   | 1                             | 642    |
| Campaign East         | 3                        | Web feature              | 56                            | Ditmars             | Email                   | 978   | 5                             | 130    |

capacity.csv

| planning_office_public_meetings_last_year | permit_office_staff_count | Capacity |
|-------------------------------------------|--------------------------|----------|
| 4                                         | 12                       | 586      |