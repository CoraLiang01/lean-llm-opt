Let $x_i$ be the scale of development per day in area $i$, where $i$ indexes the following areas:

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

Let $v_i$ be the "Value" for area $i$ (from products.csv), and $w_i$ be the "Weight" (resource requirement) for area $i$. The overall development capacity is $C = 586$ (from capacity.csv).

The model is:

$$
\max \sum_{i} v_i x_i
$$

subject to

$$
\sum_{i} w_i x_i \leq 586
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

where

\[
\begin{array}{lll}
\text{Area} & v_i~(\text{Value}) & w_i~(\text{Weight}) \\
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

All variables $x_i$ are nonnegative integers.