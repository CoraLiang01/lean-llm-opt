##### Objective Function:

$\quad \max \sum_{i \in \mathcal{A}} v_i x_i$

where:
- $\mathcal{A}$ is the set of areas (see below for full list),
- $v_i$ is the benefit (Value) per unit development in area $i$,
- $x_i$ is the scale of development per day in area $i$ (decision variable).

##### Constraints:

$\sum_{i \in \mathcal{A}} w_i x_i \leq C$

where:
- $w_i$ is the development weight (resource usage per unit) in area $i$,
- $C$ is the overall development capacity.

$x_i \geq 0 \quad \forall i \in \mathcal{A}$

##### Retrieved Information

{
  "capacity": 586,
  "areas": [
    {
      "ProductName": "Queens",
      "Value": 469,
      "Weight": 954
    },
    {
      "ProductName": "Brooklyn",
      "Value": 290,
      "Weight": 650
    },
    {
      "ProductName": "Manhattan",
      "Value": 236,
      "Weight": 961
    },
    {
      "ProductName": "Bronx",
      "Value": 235,
      "Weight": 950
    },
    {
      "ProductName": "Staten Island",
      "Value": 745,
      "Weight": 379
    },
    {
      "ProductName": "Harlem",
      "Value": 684,
      "Weight": 776
    },
    {
      "ProductName": "Upper East Side",
      "Value": 444,
      "Weight": 381
    },
    {
      "ProductName": "Lower Manhattan",
      "Value": 172,
      "Weight": 808
    },
    {
      "ProductName": "Midtown",
      "Value": 1000,
      "Weight": 937
    },
    {
      "ProductName": "Long Island City",
      "Value": 336,
      "Weight": 608
    },
    {
      "ProductName": "Williamsburg",
      "Value": 546,
      "Weight": 912
    },
    {
      "ProductName": "Bushwick",
      "Value": 535,
      "Weight": 391
    },
    {
      "ProductName": "Flatbush",
      "Value": 539,
      "Weight": 465
    },
    {
      "ProductName": "Greenpoint",
      "Value": 831,
      "Weight": 490
    },
    {
      "ProductName": "Park Slope",
      "Value": 139,
      "Weight": 918
    },
    {
      "ProductName": "Astoria",
      "Value": 432,
      "Weight": 787
    },
    {
      "ProductName": "Jackson Heights",
      "Value": 627,
      "Weight": 347
    },
    {
      "ProductName": "Flushing",
      "Value": 629,
      "Weight": 274
    },
    {
      "ProductName": "Sunnyside",
      "Value": 292,
      "Weight": 642
    },
    {
      "ProductName": "Ditmars",
      "Value": 978,
      "Weight": 130
    }
  ]
}

##### Full Model (with explicit parameters):

Let $\mathcal{A} = \{$Queens, Brooklyn, Manhattan, Bronx, Staten Island, Harlem, Upper East Side, Lower Manhattan, Midtown, Long Island City, Williamsburg, Bushwick, Flatbush, Greenpoint, Park Slope, Astoria, Jackson Heights, Flushing, Sunnyside, Ditmars$\}$.

Let $v_i$ and $w_i$ be as follows:

| Area               | $v_i$ (Value) | $w_i$ (Weight) |
|--------------------|--------------|---------------|
| Queens             | 469          | 954           |
| Brooklyn           | 290          | 650           |
| Manhattan          | 236          | 961           |
| Bronx              | 235          | 950           |
| Staten Island      | 745          | 379           |
| Harlem             | 684          | 776           |
| Upper East Side    | 444          | 381           |
| Lower Manhattan    | 172          | 808           |
| Midtown            | 1000         | 937           |
| Long Island City   | 336          | 608           |
| Williamsburg       | 546          | 912           |
| Bushwick           | 535          | 391           |
| Flatbush           | 539          | 465           |
| Greenpoint         | 831          | 490           |
| Park Slope         | 139          | 918           |
| Astoria            | 432          | 787           |
| Jackson Heights    | 627          | 347           |
| Flushing           | 629          | 274           |
| Sunnyside          | 292          | 642           |
| Ditmars            | 978          | 130           |

Overall development capacity: $C = 586$

##### Decision variables:

$x_i \geq 0$ for each area $i \in \mathcal{A}$

##### Complete Model:

$\max \Big($
$469x_{\text{Queens}} + 290x_{\text{Brooklyn}} + 236x_{\text{Manhattan}} + 235x_{\text{Bronx}} + 745x_{\text{Staten Island}} + 684x_{\text{Harlem}} + 444x_{\text{Upper East Side}} + 172x_{\text{Lower Manhattan}} + 1000x_{\text{Midtown}} + 336x_{\text{Long Island City}} + 546x_{\text{Williamsburg}} + 535x_{\text{Bushwick}} + 539x_{\text{Flatbush}} + 831x_{\text{Greenpoint}} + 139x_{\text{Park Slope}} + 432x_{\text{Astoria}} + 627x_{\text{Jackson Heights}} + 629x_{\text{Flushing}} + 292x_{\text{Sunnyside}} + 978x_{\text{Ditmars}}$
$\Big)$

subject to

$954x_{\text{Queens}} + 650x_{\text{Brooklyn}} + 961x_{\text{Manhattan}} + 950x_{\text{Bronx}} + 379x_{\text{Staten Island}} + 776x_{\text{Harlem}} + 381x_{\text{Upper East Side}} + 808x_{\text{Lower Manhattan}} + 937x_{\text{Midtown}} + 608x_{\text{Long Island City}} + 912x_{\text{Williamsburg}} + 391x_{\text{Bushwick}} + 465x_{\text{Flatbush}} + 490x_{\text{Greenpoint}} + 918x_{\text{Park Slope}} + 787x_{\text{Astoria}} + 347x_{\text{Jackson Heights}} + 274x_{\text{Flushing}} + 642x_{\text{Sunnyside}} + 130x_{\text{Ditmars}} \leq 586$

and

$x_i \geq 0 \quad \forall i \in \mathcal{A}$