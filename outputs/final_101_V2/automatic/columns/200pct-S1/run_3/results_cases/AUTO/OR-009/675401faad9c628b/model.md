##### Sets and Indices

Let $I$ be the set of areas (from ProductName in products.csv):
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

Let $x_i$ = scale of development per day in area $i$, for each $i \in I$.

##### Parameters

For each area $i$:
- $v_i$ = Value (development benefit) of area $i$ (from products.csv):

\[
\begin{align*}
&\text{Queens:} & v_{\text{Queens}} &= 469 \\
&\text{Brooklyn:} & v_{\text{Brooklyn}} &= 290 \\
&\text{Manhattan:} & v_{\text{Manhattan}} &= 236 \\
&\text{Bronx:} & v_{\text{Bronx}} &= 235 \\
&\text{Staten Island:} & v_{\text{Staten Island}} &= 745 \\
&\text{Harlem:} & v_{\text{Harlem}} &= 684 \\
&\text{Upper East Side:} & v_{\text{Upper East Side}} &= 444 \\
&\text{Lower Manhattan:} & v_{\text{Lower Manhattan}} &= 172 \\
&\text{Midtown:} & v_{\text{Midtown}} &= 1000 \\
&\text{Long Island City:} & v_{\text{Long Island City}} &= 336 \\
&\text{Williamsburg:} & v_{\text{Williamsburg}} &= 546 \\
&\text{Bushwick:} & v_{\text{Bushwick}} &= 535 \\
&\text{Flatbush:} & v_{\text{Flatbush}} &= 539 \\
&\text{Greenpoint:} & v_{\text{Greenpoint}} &= 831 \\
&\text{Park Slope:} & v_{\text{Park Slope}} &= 139 \\
&\text{Astoria:} & v_{\text{Astoria}} &= 432 \\
&\text{Jackson Heights:} & v_{\text{Jackson Heights}} &= 627 \\
&\text{Flushing:} & v_{\text{Flushing}} &= 629 \\
&\text{Sunnyside:} & v_{\text{Sunnyside}} &= 292 \\
&\text{Ditmars:} & v_{\text{Ditmars}} &= 978 \\
\end{align*}
\]

Let $w_i$ = Weight (resource consumption per unit development in area $i$):

\[
\begin{align*}
&\text{Queens:} & w_{\text{Queens}} &= 954 \\
&\text{Brooklyn:} & w_{\text{Brooklyn}} &= 650 \\
&\text{Manhattan:} & w_{\text{Manhattan}} &= 961 \\
&\text{Bronx:} & w_{\text{Bronx}} &= 950 \\
&\text{Staten Island:} & w_{\text{Staten Island}} &= 379 \\
&\text{Harlem:} & w_{\text{Harlem}} &= 776 \\
&\text{Upper East Side:} & w_{\text{Upper East Side}} &= 381 \\
&\text{Lower Manhattan:} & w_{\text{Lower Manhattan}} &= 808 \\
&\text{Midtown:} & w_{\text{Midtown}} &= 937 \\
&\text{Long Island City:} & w_{\text{Long Island City}} &= 608 \\
&\text{Williamsburg:} & w_{\text{Williamsburg}} &= 912 \\
&\text{Bushwick:} & w_{\text{Bushwick}} &= 391 \\
&\text{Flatbush:} & w_{\text{Flatbush}} &= 465 \\
&\text{Greenpoint:} & w_{\text{Greenpoint}} &= 490 \\
&\text{Park Slope:} & w_{\text{Park Slope}} &= 918 \\
&\text{Astoria:} & w_{\text{Astoria}} &= 787 \\
&\text{Jackson Heights:} & w_{\text{Jackson Heights}} &= 347 \\
&\text{Flushing:} & w_{\text{Flushing}} &= 274 \\
&\text{Sunnyside:} & w_{\text{Sunnyside}} &= 642 \\
&\text{Ditmars:} & w_{\text{Ditmars}} &= 130 \\
\end{align*}
\]

Let $C$ = overall development capacity (from capacity.csv):

\[
C = 586
\]

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
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

##### All identifiers and coefficients used above are taken directly from the provided data, in source order.