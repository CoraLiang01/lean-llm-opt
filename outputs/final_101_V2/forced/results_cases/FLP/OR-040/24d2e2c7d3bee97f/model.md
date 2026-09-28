##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: integer number of development units per day in area $i$, for each area $i \in A$.

##### Parameters

Let $A$ be the set of areas:
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

Let $w_i$ be the weight (development units per area) for area $i$:
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
- $w_{\text{Williamsburg}} = 387$
- $w_{\text{Bushwick}} = 425$
- $w_{\text{Flatbush}} = 482$
- $w_{\text{Greenpoint}} = 495$
- $w_{\text{Park Slope}} = 305$
- $w_{\text{Astoria}} = 377$
- $w_{\text{Jackson Heights}} = 318$
- $w_{\text{Flushing}} = 56$
- $w_{\text{Sunnyside}} = 213$
- $w_{\text{Ditmars}} = 472$

Total development capacity: $C = 4466$

##### Objective Function

\[
\max \sum_{i \in A} v_i x_i
\]

##### Constraints

1. Capacity constraint:
   \[
   \sum_{i \in A} w_i x_i \leq 4466
   \]
2. Integer and nonnegativity constraints:
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in A
   \]

##### Complete Model

\[
\begin{align*}
\max\ & 443x_{\text{Queens}} + 522x_{\text{Brooklyn}} + 300x_{\text{Manhattan}} + 767x_{\text{Bronx}} + 300x_{\text{Staten Island}} + 309x_{\text{Harlem}} + 598x_{\text{Upper East Side}} \\
& + 460x_{\text{Lower Manhattan}} + 318x_{\text{Midtown}} + 126x_{\text{Long Island City}} + 593x_{\text{Williamsburg}} + 871x_{\text{Bushwick}} + 858x_{\text{Flatbush}} \\
& + 321x_{\text{Greenpoint}} + 275x_{\text{Park Slope}} + 700x_{\text{Astoria}} + 685x_{\text{Jackson Heights}} + 940x_{\text{Flushing}} + 522x_{\text{Sunnyside}} + 763x_{\text{Ditmars}} \\
\text{s.t.}\quad & 104x_{\text{Queens}} + 368x_{\text{Brooklyn}} + 483x_{\text{Manhattan}} + 165x_{\text{Bronx}} + 105x_{\text{Staten Island}} + 123x_{\text{Harlem}} + 131x_{\text{Upper East Side}} \\
& + 341x_{\text{Lower Manhattan}} + 258x_{\text{Midtown}} + 469x_{\text{Long Island City}} + 387x_{\text{Williamsburg}} + 425x_{\text{Bushwick}} + 482x_{\text{Flatbush}} \\
& + 495x_{\text{Greenpoint}} + 305x_{\text{Park Slope}} + 377x_{\text{Astoria}} + 318x_{\text{Jackson Heights}} + 56x_{\text{Flushing}} + 213x_{\text{Sunnyside}} + 472x_{\text{Ditmars}} \leq 4466 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in A
\end{align*}
\]