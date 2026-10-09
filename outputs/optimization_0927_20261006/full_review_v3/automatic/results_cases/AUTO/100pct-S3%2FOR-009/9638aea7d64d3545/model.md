##### Sets and Indices

Let $I$ be the set of areas (indexed by $i$), with the following elements in source order:
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
- Astoria
- Jackson Heights
- Flushing
- Sunnyside
- Ditmars

##### Parameters

For each area $i \in I$ (in source order):

| Area              | Value ($v_i$) | Weight ($w_i$) |
|-------------------|--------------|---------------|
| Queens            | 469          | 954           |
| Brooklyn          | 290          | 650           |
| Manhattan         | 236          | 961           |
| Bronx             | 235          | 950           |
| Staten Island     | 745          | 379           |
| Harlem            | 684          | 776           |
| Upper East Side   | 444          | 381           |
| Lower Manhattan   | 172          | 808           |
| Midtown           | 1000         | 937           |
| Long Island City  | 336          | 608           |
| Williamsburg      | 546          | 912           |
| Bushwick          | 535          | 391           |
| Flatbush          | 539          | 465           |
| Greenpoint        | 831          | 490           |
| Astoria           | 432          | 787           |
| Jackson Heights   | 627          | 347           |
| Flushing          | 629          | 274           |
| Sunnyside         | 292          | 642           |
| Ditmars           | 978          | 130           |

Let $C = 586$ be the overall development capacity.

##### Decision Variables

For each area $i \in I$:
- $x_i$: scale of development per day in area $i$ (nonnegative integer)

##### Mathematical Model

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]
where $v_i$ is the Value for area $i$.

**Constraint:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
where $w_i$ is the Weight for area $i$, and $C = 586$.

**Variable Domains:**
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]

##### Complete Formulation (with all coefficients):

\[
\begin{align*}
\max \quad & 469x_{\text{Queens}} + 290x_{\text{Brooklyn}} + 236x_{\text{Manhattan}} + 235x_{\text{Bronx}} + 745x_{\text{Staten Island}} + 684x_{\text{Harlem}} + 444x_{\text{Upper East Side}} \\
& + 172x_{\text{Lower Manhattan}} + 1000x_{\text{Midtown}} + 336x_{\text{Long Island City}} + 546x_{\text{Williamsburg}} + 535x_{\text{Bushwick}} + 539x_{\text{Flatbush}} \\
& + 831x_{\text{Greenpoint}} + 432x_{\text{Astoria}} + 627x_{\text{Jackson Heights}} + 629x_{\text{Flushing}} + 292x_{\text{Sunnyside}} + 978x_{\text{Ditmars}} \\
\text{s.t.} \quad & 954x_{\text{Queens}} + 650x_{\text{Brooklyn}} + 961x_{\text{Manhattan}} + 950x_{\text{Bronx}} + 379x_{\text{Staten Island}} + 776x_{\text{Harlem}} + 381x_{\text{Upper East Side}} \\
& + 808x_{\text{Lower Manhattan}} + 937x_{\text{Midtown}} + 608x_{\text{Long Island City}} + 912x_{\text{Williamsburg}} + 391x_{\text{Bushwick}} + 465x_{\text{Flatbush}} \\
& + 490x_{\text{Greenpoint}} + 787x_{\text{Astoria}} + 347x_{\text{Jackson Heights}} + 274x_{\text{Flushing}} + 642x_{\text{Sunnyside}} + 130x_{\text{Ditmars}} \leq 586 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]