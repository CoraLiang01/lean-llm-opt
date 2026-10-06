Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the areas listed by ProductName in products.csv.

Parameters (from products.csv, in source order):

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Queens              | 469   | 954    |
| Brooklyn            | 290   | 650    |
| Manhattan           | 236   | 961    |
| Bronx               | 235   | 950    |
| Staten Island       | 745   | 379    |
| Harlem              | 684   | 776    |
| Upper East Side     | 444   | 381    |
| Lower Manhattan     | 172   | 808    |
| Midtown             | 1000  | 937    |
| Long Island City    | 336   | 608    |
| Williamsburg        | 546   | 912    |
| Bushwick            | 535   | 391    |
| Flatbush            | 539   | 465    |
| Greenpoint          | 831   | 490    |
| Park Slope          | 139   | 918    |
| Astoria             | 432   | 787    |
| Jackson Heights     | 627   | 347    |
| Flushing            | 629   | 274    |
| Sunnyside           | 292   | 642    |
| Ditmars             | 978   | 130    |

From capacity.csv:
- Total development capacity: $586$

Mathematical Model:

Objective:
$$
\max \left(
469\,x_{\text{Queens}} +
290\,x_{\text{Brooklyn}} +
236\,x_{\text{Manhattan}} +
235\,x_{\text{Bronx}} +
745\,x_{\text{Staten Island}} +
684\,x_{\text{Harlem}} +
444\,x_{\text{Upper East Side}} +
172\,x_{\text{Lower Manhattan}} +
1000\,x_{\text{Midtown}} +
336\,x_{\text{Long Island City}} +
546\,x_{\text{Williamsburg}} +
535\,x_{\text{Bushwick}} +
539\,x_{\text{Flatbush}} +
831\,x_{\text{Greenpoint}} +
139\,x_{\text{Park Slope}} +
432\,x_{\text{Astoria}} +
627\,x_{\text{Jackson Heights}} +
629\,x_{\text{Flushing}} +
292\,x_{\text{Sunnyside}} +
978\,x_{\text{Ditmars}}
\right)
$$

Subject to:

$$
954\,x_{\text{Queens}} +
650\,x_{\text{Brooklyn}} +
961\,x_{\text{Manhattan}} +
950\,x_{\text{Bronx}} +
379\,x_{\text{Staten Island}} +
776\,x_{\text{Harlem}} +
381\,x_{\text{Upper East Side}} +
808\,x_{\text{Lower Manhattan}} +
937\,x_{\text{Midtown}} +
608\,x_{\text{Long Island City}} +
912\,x_{\text{Williamsburg}} +
391\,x_{\text{Bushwick}} +
465\,x_{\text{Flatbush}} +
490\,x_{\text{Greenpoint}} +
918\,x_{\text{Park Slope}} +
787\,x_{\text{Astoria}} +
347\,x_{\text{Jackson Heights}} +
274\,x_{\text{Flushing}} +
642\,x_{\text{Sunnyside}} +
130\,x_{\text{Ditmars}}
\leq 586
$$

$$
x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in \{\text{all ProductName above, in order}\}
$$

Where:
- $x_i$ = scale of development per day in area $i$ (nonnegative integer)
- Value = development benefit per unit in area $i$
- Weight = resource consumption per unit in area $i$
- Total development capacity = $586$ units of resource

All coefficients and identifiers are as retrieved, in original order.