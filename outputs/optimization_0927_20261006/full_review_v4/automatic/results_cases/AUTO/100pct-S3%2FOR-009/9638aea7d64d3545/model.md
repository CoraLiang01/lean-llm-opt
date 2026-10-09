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

For each area $i \in I$ (in the order above):

- $v_i$ = development benefit per unit:
  - Queens: $469$
  - Brooklyn: $290$
  - Manhattan: $236$
  - Bronx: $235$
  - Staten Island: $745$
  - Harlem: $684$
  - Upper East Side: $444$
  - Lower Manhattan: $172$
  - Midtown: $1000$
  - Long Island City: $336$
  - Williamsburg: $546$
  - Bushwick: $535$
  - Flatbush: $539$
  - Greenpoint: $831$
  - Astoria: $432$
  - Jackson Heights: $627$
  - Flushing: $629$
  - Sunnyside: $292$
  - Ditmars: $978$

- $w_i$ = resource requirement per unit:
  - Queens: $954$
  - Brooklyn: $650$
  - Manhattan: $961$
  - Bronx: $950$
  - Staten Island: $379$
  - Harlem: $776$
  - Upper East Side: $381$
  - Lower Manhattan: $808$
  - Midtown: $937$
  - Long Island City: $608$
  - Williamsburg: $912$
  - Bushwick: $391$
  - Flatbush: $465$
  - Greenpoint: $490$
  - Astoria: $787$
  - Jackson Heights: $347$
  - Flushing: $274$
  - Sunnyside: $642$
  - Ditmars: $130$

Let $C = 586$ be the overall development capacity.

##### Decision Variables

For each area $i \in I$:
- $x_i \in \mathbb{Z}_{\geq 0}$: scale of development per day in area $i$

##### Mathematical Model

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]
That is,
\[
\max \bigg(
469\,x_{\text{Queens}} + 290\,x_{\text{Brooklyn}} + 236\,x_{\text{Manhattan}} + 235\,x_{\text{Bronx}} + 745\,x_{\text{Staten Island}} + 684\,x_{\text{Harlem}} + 444\,x_{\text{Upper East Side}} + 172\,x_{\text{Lower Manhattan}} + 1000\,x_{\text{Midtown}} + 336\,x_{\text{Long Island City}} + 546\,x_{\text{Williamsburg}} + 535\,x_{\text{Bushwick}} + 539\,x_{\text{Flatbush}} + 831\,x_{\text{Greenpoint}} + 432\,x_{\text{Astoria}} + 627\,x_{\text{Jackson Heights}} + 629\,x_{\text{Flushing}} + 292\,x_{\text{Sunnyside}} + 978\,x_{\text{Ditmars}}
\bigg)
\]

**Subject to:**

- Overall development capacity:
\[
\sum_{i \in I} w_i x_i \leq 586
\]
That is,
\[
954\,x_{\text{Queens}} + 650\,x_{\text{Brooklyn}} + 961\,x_{\text{Manhattan}} + 950\,x_{\text{Bronx}} + 379\,x_{\text{Staten Island}} + 776\,x_{\text{Harlem}} + 381\,x_{\text{Upper East Side}} + 808\,x_{\text{Lower Manhattan}} + 937\,x_{\text{Midtown}} + 608\,x_{\text{Long Island City}} + 912\,x_{\text{Williamsburg}} + 391\,x_{\text{Bushwick}} + 465\,x_{\text{Flatbush}} + 490\,x_{\text{Greenpoint}} + 787\,x_{\text{Astoria}} + 347\,x_{\text{Jackson Heights}} + 274\,x_{\text{Flushing}} + 642\,x_{\text{Sunnyside}} + 130\,x_{\text{Ditmars}} \leq 586
\]

- Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

##### Complete Model

\[
\begin{align*}
\max\quad & 469\,x_{\text{Queens}} + 290\,x_{\text{Brooklyn}} + 236\,x_{\text{Manhattan}} + 235\,x_{\text{Bronx}} + 745\,x_{\text{Staten Island}} + 684\,x_{\text{Harlem}} + 444\,x_{\text{Upper East Side}} + 172\,x_{\text{Lower Manhattan}} \\
& + 1000\,x_{\text{Midtown}} + 336\,x_{\text{Long Island City}} + 546\,x_{\text{Williamsburg}} + 535\,x_{\text{Bushwick}} + 539\,x_{\text{Flatbush}} + 831\,x_{\text{Greenpoint}} \\
& + 432\,x_{\text{Astoria}} + 627\,x_{\text{Jackson Heights}} + 629\,x_{\text{Flushing}} + 292\,x_{\text{Sunnyside}} + 978\,x_{\text{Ditmars}} \\
\text{s.t.}\quad & 954\,x_{\text{Queens}} + 650\,x_{\text{Brooklyn}} + 961\,x_{\text{Manhattan}} + 950\,x_{\text{Bronx}} + 379\,x_{\text{Staten Island}} + 776\,x_{\text{Harlem}} + 381\,x_{\text{Upper East Side}} \\
& + 808\,x_{\text{Lower Manhattan}} + 937\,x_{\text{Midtown}} + 608\,x_{\text{Long Island City}} + 912\,x_{\text{Williamsburg}} + 391\,x_{\text{Bushwick}} + 465\,x_{\text{Flatbush}} \\
& + 490\,x_{\text{Greenpoint}} + 787\,x_{\text{Astoria}} + 347\,x_{\text{Jackson Heights}} + 274\,x_{\text{Flushing}} + 642\,x_{\text{Sunnyside}} + 130\,x_{\text{Ditmars}} \leq 586 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\end{align*}
\]