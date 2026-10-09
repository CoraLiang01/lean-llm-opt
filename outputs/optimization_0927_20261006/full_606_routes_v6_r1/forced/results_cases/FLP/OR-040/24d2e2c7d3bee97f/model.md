##### Decision Variables

Let $x_i$ be the integer number of development units chosen daily in area $i$, for each area $i \in A$.

##### Parameters

- $A$ (Areas):  
  Queens, Brooklyn, Manhattan, Bronx, Staten Island, Harlem, Upper East Side, Lower Manhattan, Midtown, Long Island City, Williamsburg, Bushwick, Flatbush, Greenpoint, Park Slope, Astoria, Jackson Heights, Flushing, Sunnyside, Ditmars

- Benefit coefficients $v_i$ and weights $w_i$ for each area $i$:

| Area                | $v_i$ (Value) | $w_i$ (Weight) |
|---------------------|:-------------:|:--------------:|
| Queens              | 443           | 104            |
| Brooklyn            | 522           | 368            |
| Manhattan           | 300           | 483            |
| Bronx               | 767           | 165            |
| Staten Island       | 300           | 105            |
| Harlem              | 309           | 123            |
| Upper East Side     | 598           | 131            |
| Lower Manhattan     | 460           | 341            |
| Midtown             | 318           | 258            |
| Long Island City    | 126           | 469            |
| Williamsburg        | 593           | 387            |
| Bushwick            | 871           | 425            |
| Flatbush            | 858           | 482            |
| Greenpoint          | 321           | 495            |
| Park Slope          | 275           | 305            |
| Astoria             | 700           | 377            |
| Jackson Heights     | 685           | 318            |
| Flushing            | 940           | 56             |
| Sunnyside           | 522           | 213            |
| Ditmars             | 763           | 472            |

- Total development capacity: $C = 4466$

##### Mathematical Model

**Objective:**
\[
\max \sum_{i \in A} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in A} w_i x_i \leq 4466
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in A
\]

##### Retrieved Information

- Areas $A$:
  Queens, Brooklyn, Manhattan, Bronx, Staten Island, Harlem, Upper East Side, Lower Manhattan, Midtown, Long Island City, Williamsburg, Bushwick, Flatbush, Greenpoint, Park Slope, Astoria, Jackson Heights, Flushing, Sunnyside, Ditmars

- Benefit coefficients $v_i$:
  Queens: 443, Brooklyn: 522, Manhattan: 300, Bronx: 767, Staten Island: 300, Harlem: 309, Upper East Side: 598, Lower Manhattan: 460, Midtown: 318, Long Island City: 126, Williamsburg: 593, Bushwick: 871, Flatbush: 858, Greenpoint: 321, Park Slope: 275, Astoria: 700, Jackson Heights: 685, Flushing: 940, Sunnyside: 522, Ditmars: 763

- Weights $w_i$:
  Queens: 104, Brooklyn: 368, Manhattan: 483, Bronx: 165, Staten Island: 105, Harlem: 123, Upper East Side: 131, Lower Manhattan: 341, Midtown: 258, Long Island City: 469, Williamsburg: 387, Bushwick: 425, Flatbush: 482, Greenpoint: 495, Park Slope: 305, Astoria: 377, Jackson Heights: 318, Flushing: 56, Sunnyside: 213, Ditmars: 472

- Capacity $C = 4466$

##### Complete Model

\[
\begin{align*}
\max\quad & 443x_{\text{Queens}} + 522x_{\text{Brooklyn}} + 300x_{\text{Manhattan}} + 767x_{\text{Bronx}} + 300x_{\text{Staten Island}} + 309x_{\text{Harlem}} + 598x_{\text{Upper East Side}} \\
& + 460x_{\text{Lower Manhattan}} + 318x_{\text{Midtown}} + 126x_{\text{Long Island City}} + 593x_{\text{Williamsburg}} + 871x_{\text{Bushwick}} + 858x_{\text{Flatbush}} \\
& + 321x_{\text{Greenpoint}} + 275x_{\text{Park Slope}} + 700x_{\text{Astoria}} + 685x_{\text{Jackson Heights}} + 940x_{\text{Flushing}} + 522x_{\text{Sunnyside}} + 763x_{\text{Ditmars}} \\
\text{s.t.}\quad & 104x_{\text{Queens}} + 368x_{\text{Brooklyn}} + 483x_{\text{Manhattan}} + 165x_{\text{Bronx}} + 105x_{\text{Staten Island}} + 123x_{\text{Harlem}} + 131x_{\text{Upper East Side}} \\
& + 341x_{\text{Lower Manhattan}} + 258x_{\text{Midtown}} + 469x_{\text{Long Island City}} + 387x_{\text{Williamsburg}} + 425x_{\text{Bushwick}} + 482x_{\text{Flatbush}} \\
& + 495x_{\text{Greenpoint}} + 305x_{\text{Park Slope}} + 377x_{\text{Astoria}} + 318x_{\text{Jackson Heights}} + 56x_{\text{Flushing}} + 213x_{\text{Sunnyside}} + 472x_{\text{Ditmars}} \leq 4466 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in A
\end{align*}
\]