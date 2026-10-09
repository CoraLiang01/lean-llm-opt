Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas:

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

Let $v_i$ be the benefit per unit development in area $i$ (from the "Value" column), and $w_i$ be the resource requirement per unit development in area $i$ (from the "Weight" column). The total available development capacity is $586$ (from the "Capacity" column in capacity.csv).

The mathematical model is:

Objective:
$$
\max \;
469 x_{\text{Queens}}
+ 290 x_{\text{Brooklyn}}
+ 236 x_{\text{Manhattan}}
+ 235 x_{\text{Bronx}}
+ 745 x_{\text{Staten Island}}
+ 684 x_{\text{Harlem}}
+ 444 x_{\text{Upper East Side}}
+ 172 x_{\text{Lower Manhattan}}
+ 1000 x_{\text{Midtown}}
+ 336 x_{\text{Long Island City}}
+ 546 x_{\text{Williamsburg}}
+ 535 x_{\text{Bushwick}}
+ 539 x_{\text{Flatbush}}
+ 831 x_{\text{Greenpoint}}
+ 139 x_{\text{Park Slope}}
+ 432 x_{\text{Astoria}}
+ 627 x_{\text{Jackson Heights}}
+ 629 x_{\text{Flushing}}
+ 292 x_{\text{Sunnyside}}
+ 978 x_{\text{Ditmars}}
$$

Subject to:

Resource capacity constraint:
$$
954 x_{\text{Queens}}
+ 650 x_{\text{Brooklyn}}
+ 961 x_{\text{Manhattan}}
+ 950 x_{\text{Bronx}}
+ 379 x_{\text{Staten Island}}
+ 776 x_{\text{Harlem}}
+ 381 x_{\text{Upper East Side}}
+ 808 x_{\text{Lower Manhattan}}
+ 937 x_{\text{Midtown}}
+ 608 x_{\text{Long Island City}}
+ 912 x_{\text{Williamsburg}}
+ 391 x_{\text{Bushwick}}
+ 465 x_{\text{Flatbush}}
+ 490 x_{\text{Greenpoint}}
+ 918 x_{\text{Park Slope}}
+ 787 x_{\text{Astoria}}
+ 347 x_{\text{Jackson Heights}}
+ 274 x_{\text{Flushing}}
+ 642 x_{\text{Sunnyside}}
+ 130 x_{\text{Ditmars}}
\leq 586
$$

Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

Where the correspondence between $i$ and area names is as listed above, and all coefficients are as given in the retrieved data.