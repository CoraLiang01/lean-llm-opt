Let $x_{ij}$ be the number of vessels of type $j$ (ProductName $j$) to be placed in display area $i$ (DisplayID $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $D$ be the set of display areas (DisplayID):  
  $D = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14\}$

- Let $P$ be the set of vessel types (ProductName):  
  $P = \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$

- For each display area $i \in D$, let $C_i$ be its capacity:

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

- For each vessel type $j \in P$, let $v_j$ be its value and $w_j$ its weight (size):

  \[
  \begin{array}{lll}
  \text{ProductName} & v_j & w_j \\
  \hline
  \text{Speedboat} & 29664 & 18 \\
  \text{Fishing Boat} & 31778 & 36 \\
  \text{Catamaran} & 73501 & 25 \\
  \text{Yacht} & 78255 & 16 \\
  \text{Sailboat} & 93606 & 97 \\
  \text{Kayak} & 46983 & 35 \\
  \text{Canoe} & 95026 & 32 \\
  \text{Houseboat} & 57685 & 100 \\
  \text{Pontoon} & 60323 & 43 \\
  \text{Jet Ski} & 91224 & 15 \\
  \text{Rowboat} & 44003 & 95 \\
  \text{Hovercraft} & 75998 & 57 \\
  \text{Cabin Cruiser} & 84525 & 13 \\
  \text{Wakeboard Boat} & 66207 & 44 \\
  \text{Dinghy} & 65002 & 64 \\
  \text{Trawler} & 33132 & 88 \\
  \text{Paddle Boat} & 69239 & 42 \\
  \text{Submarine} & 66948 & 46 \\
  \text{RIB} & 88240 & 24 \\
  \text{Skiff} & 48858 & 93 \\
  \end{array}
  \]

---

### Mathematical Model

**Decision Variables:**

\[
x_{ij} = \text{number of vessels of type } j \text{ placed in display area } i, \quad \forall i \in D,\, j \in P
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

**Objective:**

\[
\max \sum_{i \in D} \sum_{j \in P} v_j \cdot x_{ij}
\]

**Subject to:**

For each display area $i \in D$:

\[
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
\]

For all $i \in D$, $j \in P$:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**All parameters and identifiers are as retrieved and preserved in original order.**