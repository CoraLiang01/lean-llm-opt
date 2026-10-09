##### Sets and Indices

Let $I$ be the set of areas (indexed by $i$), with the following members in source order:
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

For each area $i$:
- $v_i$ = Value (development benefit per unit)
- $w_i$ = Weight (resource consumption per unit)

From the data:
- Queens: $v_{\text{Queens}} = 469$, $w_{\text{Queens}} = 954$
- Brooklyn: $v_{\text{Brooklyn}} = 290$, $w_{\text{Brooklyn}} = 650$
- Manhattan: $v_{\text{Manhattan}} = 236$, $w_{\text{Manhattan}} = 961$
- Bronx: $v_{\text{Bronx}} = 235$, $w_{\text{Bronx}} = 950$
- Staten Island: $v_{\text{Staten Island}} = 745$, $w_{\text{Staten Island}} = 379$
- Harlem: $v_{\text{Harlem}} = 684$, $w_{\text{Harlem}} = 776$
- Upper East Side: $v_{\text{Upper East Side}} = 444$, $w_{\text{Upper East Side}} = 381$
- Lower Manhattan: $v_{\text{Lower Manhattan}} = 172$, $w_{\text{Lower Manhattan}} = 808$
- Midtown: $v_{\text{Midtown}} = 1000$, $w_{\text{Midtown}} = 937$
- Long Island City: $v_{\text{Long Island City}} = 336$, $w_{\text{Long Island City}} = 608$
- Williamsburg: $v_{\text{Williamsburg}} = 546$, $w_{\text{Williamsburg}} = 912$
- Bushwick: $v_{\text{Bushwick}} = 535$, $w_{\text{Bushwick}} = 391$
- Flatbush: $v_{\text{Flatbush}} = 539$, $w_{\text{Flatbush}} = 465$
- Greenpoint: $v_{\text{Greenpoint}} = 831$, $w_{\text{Greenpoint}} = 490$
- Park Slope: $v_{\text{Park Slope}} = 139$, $w_{\text{Park Slope}} = 918$
- Astoria: $v_{\text{Astoria}} = 432$, $w_{\text{Astoria}} = 787$
- Jackson Heights: $v_{\text{Jackson Heights}} = 627$, $w_{\text{Jackson Heights}} = 347$
- Flushing: $v_{\text{Flushing}} = 629$, $w_{\text{Flushing}} = 274$
- Sunnyside: $v_{\text{Sunnyside}} = 292$, $w_{\text{Sunnyside}} = 642$
- Ditmars: $v_{\text{Ditmars}} = 978$, $w_{\text{Ditmars}} = 130$

Total development capacity: $C = 586$

##### Decision Variables

For each area $i$:
- $x_i \geq 0$ (continuous), representing the scale of development per day in area $i$.

##### Mathematical Model

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq 586
\]
\[
x_i \geq 0 \quad \forall i \in I
\]

##### Data Table (source order)

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

**Capacity:** $586$

##### Complete Model

\[
\begin{align*}
\max\quad & 469x_{\text{Queens}} + 290x_{\text{Brooklyn}} + 236x_{\text{Manhattan}} + 235x_{\text{Bronx}} + 745x_{\text{Staten Island}} + 684x_{\text{Harlem}} \\
& + 444x_{\text{Upper East Side}} + 172x_{\text{Lower Manhattan}} + 1000x_{\text{Midtown}} + 336x_{\text{Long Island City}} + 546x_{\text{Williamsburg}} \\
& + 535x_{\text{Bushwick}} + 539x_{\text{Flatbush}} + 831x_{\text{Greenpoint}} + 139x_{\text{Park Slope}} + 432x_{\text{Astoria}} \\
& + 627x_{\text{Jackson Heights}} + 629x_{\text{Flushing}} + 292x_{\text{Sunnyside}} + 978x_{\text{Ditmars}} \\
\text{s.t.}\quad & 954x_{\text{Queens}} + 650x_{\text{Brooklyn}} + 961x_{\text{Manhattan}} + 950x_{\text{Bronx}} + 379x_{\text{Staten Island}} + 776x_{\text{Harlem}} \\
& + 381x_{\text{Upper East Side}} + 808x_{\text{Lower Manhattan}} + 937x_{\text{Midtown}} + 608x_{\text{Long Island City}} + 912x_{\text{Williamsburg}} \\
& + 391x_{\text{Bushwick}} + 465x_{\text{Flatbush}} + 490x_{\text{Greenpoint}} + 918x_{\text{Park Slope}} + 787x_{\text{Astoria}} \\
& + 347x_{\text{Jackson Heights}} + 274x_{\text{Flushing}} + 642x_{\text{Sunnyside}} + 130x_{\text{Ditmars}} \leq 586 \\
& x_i \geq 0 \quad \forall i \in I
\end{align*}
\]