##### Objective Function:

$\quad \max \sum_{i=1}^{20} v_i x_i$

where $x_i$ is the scale of development per day in area $i$, and $v_i$ is the benefit per unit scale in area $i$.

##### Constraints:

$\sum_{i=1}^{20} w_i x_i \leq 586$

$x_i \geq 0 \quad \forall i \in \{1,2,\ldots,20\}$

##### Retrieved Information

{
  "areas": [
    "Queens",
    "Brooklyn",
    "Manhattan",
    "Bronx",
    "Staten Island",
    "Harlem",
    "Upper East Side",
    "Lower Manhattan",
    "Midtown",
    "Long Island City",
    "Williamsburg",
    "Bushwick",
    "Flatbush",
    "Greenpoint",
    "Park Slope",
    "Astoria",
    "Jackson Heights",
    "Flushing",
    "Sunnyside",
    "Ditmars"
  ],
  "benefit": {
    "Queens": 469,
    "Brooklyn": 290,
    "Manhattan": 236,
    "Bronx": 235,
    "Staten Island": 745,
    "Harlem": 684,
    "Upper East Side": 444,
    "Lower Manhattan": 172,
    "Midtown": 1000,
    "Long Island City": 336,
    "Williamsburg": 546,
    "Bushwick": 535,
    "Flatbush": 539,
    "Greenpoint": 831,
    "Park Slope": 139,
    "Astoria": 432,
    "Jackson Heights": 627,
    "Flushing": 629,
    "Sunnyside": 292,
    "Ditmars": 978
  },
  "weight": {
    "Queens": 954,
    "Brooklyn": 650,
    "Manhattan": 961,
    "Bronx": 950,
    "Staten Island": 379,
    "Harlem": 776,
    "Upper East Side": 381,
    "Lower Manhattan": 808,
    "Midtown": 937,
    "Long Island City": 608,
    "Williamsburg": 912,
    "Bushwick": 391,
    "Flatbush": 465,
    "Greenpoint": 490,
    "Park Slope": 918,
    "Astoria": 787,
    "Jackson Heights": 347,
    "Flushing": 274,
    "Sunnyside": 642,
    "Ditmars": 130
  },
  "capacity": 586
}

##### Full Model (with explicit coefficients):

Let $x_i$ denote the scale of development per day in area $i$ for $i=1,\ldots,20$ (areas listed in source order).

$\max \Big($
$469x_1 + 290x_2 + 236x_3 + 235x_4 + 745x_5 + 684x_6 + 444x_7 + 172x_8 + 1000x_9 + 336x_{10} + 546x_{11} + 535x_{12} + 539x_{13} + 831x_{14} + 139x_{15} + 432x_{16} + 627x_{17} + 629x_{18} + 292x_{19} + 978x_{20}$
$\Big)$

subject to

$954x_1 + 650x_2 + 961x_3 + 950x_4 + 379x_5 + 776x_6 + 381x_7 + 808x_8 + 937x_9 + 608x_{10} + 912x_{11} + 391x_{12} + 465x_{13} + 490x_{14} + 918x_{15} + 787x_{16} + 347x_{17} + 274x_{18} + 642x_{19} + 130x_{20} \leq 586$

$x_i \geq 0 \quad \forall i=1,\ldots,20$