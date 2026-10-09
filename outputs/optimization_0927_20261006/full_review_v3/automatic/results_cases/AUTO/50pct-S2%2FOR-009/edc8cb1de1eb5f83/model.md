Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas (ProductName):

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

Let $v_i$ be the Value (development benefit) and $w_i$ be the Weight (resource requirement) for area $i$.

The overall development capacity is $586$.

The mathematical model is:

Objective:
\[
\max \; 469x_{\text{Queens}} + 290x_{\text{Brooklyn}} + 236x_{\text{Manhattan}} + 235x_{\text{Bronx}} + 745x_{\text{Staten Island}} + 684x_{\text{Harlem}} + 444x_{\text{Upper East Side}} + 172x_{\text{Lower Manhattan}} + 1000x_{\text{Midtown}} + 336x_{\text{Long Island City}} + 546x_{\text{Williamsburg}} + 535x_{\text{Bushwick}} + 539x_{\text{Flatbush}} + 831x_{\text{Greenpoint}} + 139x_{\text{Park Slope}} + 432x_{\text{Astoria}} + 627x_{\text{Jackson Heights}} + 629x_{\text{Flushing}} + 292x_{\text{Sunnyside}} + 978x_{\text{Ditmars}}
\]

Subject to:
\[
954x_{\text{Queens}} + 650x_{\text{Brooklyn}} + 961x_{\text{Manhattan}} + 950x_{\text{Bronx}} + 379x_{\text{Staten Island}} + 776x_{\text{Harlem}} + 381x_{\text{Upper East Side}} + 808x_{\text{Lower Manhattan}} + 937x_{\text{Midtown}} + 608x_{\text{Long Island City}} + 912x_{\text{Williamsburg}} + 391x_{\text{Bushwick}} + 465x_{\text{Flatbush}} + 490x_{\text{Greenpoint}} + 918x_{\text{Park Slope}} + 787x_{\text{Astoria}} + 347x_{\text{Jackson Heights}} + 274x_{\text{Flushing}} + 642x_{\text{Sunnyside}} + 130x_{\text{Ditmars}} \leq 586
\]

\[
x_i \geq 0, \quad \forall i \in \{\text{all areas listed above}\}
\]

where all $x_i$ are continuous and nonnegative (since "scale of development per day" is not explicitly required to be integer).

All coefficients and area names are as retrieved from the data.