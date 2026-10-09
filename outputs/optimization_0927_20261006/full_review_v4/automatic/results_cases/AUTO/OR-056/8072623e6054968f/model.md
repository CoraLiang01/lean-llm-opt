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
  \begin{align*}
  &\text{Speedboat:} \quad v = 29664, \quad w = 18 \\
  &\text{Fishing Boat:} \quad v = 31778, \quad w = 36 \\
  &\text{Catamaran:} \quad v = 73501, \quad w = 25 \\
  &\text{Yacht:} \quad v = 78255, \quad w = 16 \\
  &\text{Sailboat:} \quad v = 93606, \quad w = 97 \\
  &\text{Kayak:} \quad v = 46983, \quad w = 35 \\
  &\text{Canoe:} \quad v = 95026, \quad w = 32 \\
  &\text{Houseboat:} \quad v = 57685, \quad w = 100 \\
  &\text{Pontoon:} \quad v = 60323, \quad w = 43 \\
  &\text{Jet Ski:} \quad v = 91224, \quad w = 15 \\
  &\text{Rowboat:} \quad v = 44003, \quad w = 95 \\
  &\text{Hovercraft:} \quad v = 75998, \quad w = 57 \\
  &\text{Cabin Cruiser:} \quad v = 84525, \quad w = 13 \\
  &\text{Wakeboard Boat:} \quad v = 66207, \quad w = 44 \\
  &\text{Dinghy:} \quad v = 65002, \quad w = 64 \\
  &\text{Trawler:} \quad v = 33132, \quad w = 88 \\
  &\text{Paddle Boat:} \quad v = 69239, \quad w = 42 \\
  &\text{Submarine:} \quad v = 66948, \quad w = 46 \\
  &\text{RIB:} \quad v = 88240, \quad w = 24 \\
  &\text{Skiff:} \quad v = 48858, \quad w = 93 \\
  \end{align*}
  \]

---

### Mathematical Model

**Decision Variables:**

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in D, \forall j \in P
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
x_{ij} \geq 0 \quad \text{and integer}
\]

---

**All identifiers and coefficients are as retrieved and shown above.**