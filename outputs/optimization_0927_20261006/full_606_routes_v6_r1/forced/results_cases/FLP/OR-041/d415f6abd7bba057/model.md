##### Decision Variables

$x_i \geq 0$: scale of development per day in area $i$, for each area $i \in A$.

##### Parameters

Let $A$ be the set of areas:
\[
A = \{\text{Queens},\ \text{Brooklyn},\ \text{Manhattan},\ \text{Bronx},\ \text{Staten Island},\ \text{Harlem},\ \text{Upper East Side},\ \text{Lower Manhattan},\ \text{Midtown},\ \text{Long Island City},\ \text{Williamsburg},\ \text{Bushwick},\ \text{Flatbush},\ \text{Greenpoint},\ \text{Park Slope},\ \text{Astoria},\ \text{Jackson Heights},\ \text{Flushing},\ \text{Sunnyside},\ \text{Ditmars}\}
\]

For each area $i \in A$:
- $v_i$: development benefit per unit in area $i$
- $w_i$: development resource usage per unit in area $i$

The data is:

| Area                | $v_i$ (Value) | $w_i$ (Weight) |
|---------------------|:-------------:|:--------------:|
| Queens              | 469           | 954            |
| Brooklyn            | 290           | 650            |
| Manhattan           | 236           | 961            |
| Bronx               | 235           | 950            |
| Staten Island       | 745           | 379            |
| Harlem              | 684           | 776            |
| Upper East Side     | 444           | 381            |
| Lower Manhattan     | 172           | 808            |
| Midtown             | 1000          | 937            |
| Long Island City    | 336           | 608            |
| Williamsburg        | 546           | 912            |
| Bushwick            | 535           | 391            |
| Flatbush            | 539           | 465            |
| Greenpoint          | 831           | 490            |
| Park Slope          | 139           | 918            |
| Astoria             | 432           | 787            |
| Jackson Heights     | 627           | 347            |
| Flushing            | 629           | 274            |
| Sunnyside           | 292           | 642            |
| Ditmars             | 978           | 130            |

Total development capacity: $C = 586$

##### Objective Function

\[
\max \sum_{i \in A} v_i x_i
\]

##### Constraints

1. Overall development capacity:
   \[
   \sum_{i \in A} w_i x_i \leq C
   \]
2. Nonnegativity:
   \[
   x_i \geq 0 \qquad \forall i \in A
   \]

##### Full Model

\[
\begin{align*}
\max\quad & 469x_{\text{Queens}} + 290x_{\text{Brooklyn}} + 236x_{\text{Manhattan}} + 235x_{\text{Bronx}} + 745x_{\text{Staten Island}} + 684x_{\text{Harlem}} + 444x_{\text{Upper East Side}} \\
& + 172x_{\text{Lower Manhattan}} + 1000x_{\text{Midtown}} + 336x_{\text{Long Island City}} + 546x_{\text{Williamsburg}} + 535x_{\text{Bushwick}} + 539x_{\text{Flatbush}} \\
& + 831x_{\text{Greenpoint}} + 139x_{\text{Park Slope}} + 432x_{\text{Astoria}} + 627x_{\text{Jackson Heights}} + 629x_{\text{Flushing}} + 292x_{\text{Sunnyside}} + 978x_{\text{Ditmars}} \\
\text{s.t.}\quad & 954x_{\text{Queens}} + 650x_{\text{Brooklyn}} + 961x_{\text{Manhattan}} + 950x_{\text{Bronx}} + 379x_{\text{Staten Island}} + 776x_{\text{Harlem}} + 381x_{\text{Upper East Side}} \\
& + 808x_{\text{Lower Manhattan}} + 937x_{\text{Midtown}} + 608x_{\text{Long Island City}} + 912x_{\text{Williamsburg}} + 391x_{\text{Bushwick}} + 465x_{\text{Flatbush}} \\
& + 490x_{\text{Greenpoint}} + 918x_{\text{Park Slope}} + 787x_{\text{Astoria}} + 347x_{\text{Jackson Heights}} + 274x_{\text{Flushing}} + 642x_{\text{Sunnyside}} + 130x_{\text{Ditmars}} \leq 586 \\
& x_i \geq 0 \qquad \forall i \in A
\end{align*}
\]

###### Retrieved Information

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
  "value": {
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