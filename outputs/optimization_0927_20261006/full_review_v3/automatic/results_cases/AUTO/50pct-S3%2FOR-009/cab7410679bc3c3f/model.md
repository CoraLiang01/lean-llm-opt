Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas (ProductName):

\[
\begin{align*}
&\text{Queens} \\
&\text{Brooklyn} \\
&\text{Manhattan} \\
&\text{Bronx} \\
&\text{Staten Island} \\
&\text{Harlem} \\
&\text{Upper East Side} \\
&\text{Lower Manhattan} \\
&\text{Midtown} \\
&\text{Long Island City} \\
&\text{Williamsburg} \\
&\text{Bushwick} \\
&\text{Flatbush} \\
&\text{Greenpoint} \\
&\text{Park Slope} \\
&\text{Astoria} \\
&\text{Jackson Heights} \\
&\text{Flushing} \\
&\text{Sunnyside} \\
&\text{Ditmars}
\end{align*}
\]

Parameters:

- $v_i$: Value (development benefit) for area $i$ (from "Value" column)
- $w_i$: Weight (resource consumption per unit development) for area $i$ (from "Weight" column)
- $C$: Overall development capacity (from "Capacity" column in capacity.csv; $C = 586$)

Numerical values:

\[
\begin{array}{lll}
\text{Area} & v_i & w_i \\
\hline
\text{Queens} & 469 & 954 \\
\text{Brooklyn} & 290 & 650 \\
\text{Manhattan} & 236 & 961 \\
\text{Bronx} & 235 & 950 \\
\text{Staten Island} & 745 & 379 \\
\text{Harlem} & 684 & 776 \\
\text{Upper East Side} & 444 & 381 \\
\text{Lower Manhattan} & 172 & 808 \\
\text{Midtown} & 1000 & 937 \\
\text{Long Island City} & 336 & 608 \\
\text{Williamsburg} & 546 & 912 \\
\text{Bushwick} & 535 & 391 \\
\text{Flatbush} & 539 & 465 \\
\text{Greenpoint} & 831 & 490 \\
\text{Park Slope} & 139 & 918 \\
\text{Astoria} & 432 & 787 \\
\text{Jackson Heights} & 627 & 347 \\
\text{Flushing} & 629 & 274 \\
\text{Sunnyside} & 292 & 642 \\
\text{Ditmars} & 978 & 130 \\
\end{array}
\]

Model:

**Objective:**
\[
\max \sum_{i} v_i x_i
\]

**Subject to:**

\[
\sum_{i} w_i x_i \leq 586
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

**Where:**

- $x_i$ = scale of development per day in area $i$ (nonnegative integer)
- $v_i$ = Value for area $i$ (see table above)
- $w_i$ = Weight for area $i$ (see table above)
- $586$ = overall development capacity

**All data used in the order and with the identifiers as retrieved.**