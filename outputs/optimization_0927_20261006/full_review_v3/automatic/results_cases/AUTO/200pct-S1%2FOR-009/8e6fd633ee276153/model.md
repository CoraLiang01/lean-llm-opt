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

Let $v_i$ be the Value and $w_i$ be the Weight for area $i$ as given below.

The overall development capacity is $586$.

The mathematical model is:

Objective:
\[
\max \;
469 x_{\text{Queens}}
+ 290 x_{\text{Brooklyn}}
+ 236 x_{\text{Manhattan}}
+ 235 x_{\text{Bronx}}
+ 745 x_{\text{Staten Island}}
+ 684 x_{\text{Harlem}}
+ 444 x_{\text{Upper East Side}}
+ 172 x_{\text{Lower Manhattan}}
+ 1000 x_{\text{Midtown}}
+ 336 x_{\text{Long Island City}}
+ 546 x_{\text{Williamsburg}}
+ 535 x_{\text{Bushwick}}
+ 539 x_{\text{Flatbush}}
+ 831 x_{\text{Greenpoint}}
+ 139 x_{\text{Park Slope}}
+ 432 x_{\text{Astoria}}
+ 627 x_{\text{Jackson Heights}}
+ 629 x_{\text{Flushing}}
+ 292 x_{\text{Sunnyside}}
+ 978 x_{\text{Ditmars}}
\]

Subject to:

\[
954 x_{\text{Queens}}
+ 650 x_{\text{Brooklyn}}
+ 961 x_{\text{Manhattan}}
+ 950 x_{\text{Bronx}}
+ 379 x_{\text{Staten Island}}
+ 776 x_{\text{Harlem}}
+ 381 x_{\text{Upper East Side}}
+ 808 x_{\text{Lower Manhattan}}
+ 937 x_{\text{Midtown}}
+ 608 x_{\text{Long Island City}}
+ 912 x_{\text{Williamsburg}}
+ 391 x_{\text{Bushwick}}
+ 465 x_{\text{Flatbush}}
+ 490 x_{\text{Greenpoint}}
+ 918 x_{\text{Park Slope}}
+ 787 x_{\text{Astoria}}
+ 347 x_{\text{Jackson Heights}}
+ 274 x_{\text{Flushing}}
+ 642 x_{\text{Sunnyside}}
+ 130 x_{\text{Ditmars}}
\leq 586
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

Where:

\[
\begin{array}{l|r|r}
\text{Area} & \text{Value } (v_i) & \text{Weight } (w_i) \\
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

Capacity: $586$ (from capacity.csv, column "Capacity")

Decision variables: $x_i$ are nonnegative integers for all areas $i$.