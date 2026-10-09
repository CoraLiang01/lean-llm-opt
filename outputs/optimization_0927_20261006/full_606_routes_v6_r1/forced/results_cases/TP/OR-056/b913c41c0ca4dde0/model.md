##### Sets

Let $I = \{1,2,\ldots,14\}$ be the set of display areas (DisplayID from capacity.csv).

Let $J =$ 
{Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff} (ProductName from products.csv).

##### Parameters

For each display area $i \in I$:

- $C_i$ = capacity of display area $i$.

From capacity.csv (in source order):

\[
\begin{align*}
C_1 &= 457 \\
C_2 &= 604 \\
C_3 &= 751 \\
C_4 &= 468 \\
C_5 &= 343 \\
C_6 &= 408 \\
C_7 &= 741 \\
C_8 &= 914 \\
C_9 &= 682 \\
C_{10} &= 409 \\
C_{11} &= 342 \\
C_{12} &= 903 \\
C_{13} &= 680 \\
C_{14} &= 886 \\
\end{align*}
\]

For each vessel type $j \in J$:

- $v_j$ = value of vessel type $j$
- $w_j$ = weight (size) of vessel type $j$

From products.csv (in source order):

\[
\begin{align*}
&\text{Speedboat:} & v_{\text{Speedboat}} &= 29664, & w_{\text{Speedboat}} &= 18 \\
&\text{Fishing Boat:} & v_{\text{Fishing Boat}} &= 31778, & w_{\text{Fishing Boat}} &= 36 \\
&\text{Catamaran:} & v_{\text{Catamaran}} &= 73501, & w_{\text{Catamaran}} &= 25 \\
&\text{Yacht:} & v_{\text{Yacht}} &= 78255, & w_{\text{Yacht}} &= 16 \\
&\text{Sailboat:} & v_{\text{Sailboat}} &= 93606, & w_{\text{Sailboat}} &= 97 \\
&\text{Kayak:} & v_{\text{Kayak}} &= 46983, & w_{\text{Kayak}} &= 35 \\
&\text{Canoe:} & v_{\text{Canoe}} &= 95026, & w_{\text{Canoe}} &= 32 \\
&\text{Houseboat:} & v_{\text{Houseboat}} &= 57685, & w_{\text{Houseboat}} &= 100 \\
&\text{Pontoon:} & v_{\text{Pontoon}} &= 60323, & w_{\text{Pontoon}} &= 43 \\
&\text{Jet Ski:} & v_{\text{Jet Ski}} &= 91224, & w_{\text{Jet Ski}} &= 15 \\
&\text{Rowboat:} & v_{\text{Rowboat}} &= 44003, & w_{\text{Rowboat}} &= 95 \\
&\text{Hovercraft:} & v_{\text{Hovercraft}} &= 75998, & w_{\text{Hovercraft}} &= 57 \\
&\text{Cabin Cruiser:} & v_{\text{Cabin Cruiser}} &= 84525, & w_{\text{Cabin Cruiser}} &= 13 \\
&\text{Wakeboard Boat:} & v_{\text{Wakeboard Boat}} &= 66207, & w_{\text{Wakeboard Boat}} &= 44 \\
&\text{Dinghy:} & v_{\text{Dinghy}} &= 65002, & w_{\text{Dinghy}} &= 64 \\
&\text{Trawler:} & v_{\text{Trawler}} &= 33132, & w_{\text{Trawler}} &= 88 \\
&\text{Paddle Boat:} & v_{\text{Paddle Boat}} &= 69239, & w_{\text{Paddle Boat}} &= 42 \\
&\text{Submarine:} & v_{\text{Submarine}} &= 66948, & w_{\text{Submarine}} &= 46 \\
&\text{RIB:} & v_{\text{RIB}} &= 88240, & w_{\text{RIB}} &= 24 \\
&\text{Skiff:} & v_{\text{Skiff}} &= 48858, & w_{\text{Skiff}} &= 93 \\
\end{align*}
\]

##### Decision Variables

For each display area $i \in I$ and vessel type $j \in J$:

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of vessels of type $j$ placed in display area $i$

##### Objective

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

For each display area $i \in I$:

\[
\sum_{j \in J} w_j x_{ij} \leq C_i
\]

For all $i \in I$, $j \in J$:

\[
x_{ij} \geq 0 \quad \text{and integer}
\]

##### Full Model

\[
\begin{align*}
\max\quad & \sum_{i=1}^{14} \sum_{j \in J} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j \in J} w_j x_{ij} \leq C_i, \quad \forall i=1,\ldots,14 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,14,\ j \in J
\end{align*}
\]

Where:

- $C_i$ as listed above for $i=1,\ldots,14$
- $J$ and $(v_j, w_j)$ as listed above for each vessel type $j$