Let $x_{ij}$ be the number of units of boat type $j$ (ProductName) to be placed in display area $i$ (DisplayID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $D$ be the set of display areas (DisplayID):  
  $D = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14\}$

- Let $B$ be the set of boat types (ProductName):  
  $B = \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$

- For each display area $i \in D$, let $C_i$ be its capacity:

  \[
  \begin{align*}
  C_1 &= 356 \\
  C_2 &= 478 \\
  C_3 &= 305 \\
  C_4 &= 291 \\
  C_5 &= 168 \\
  C_6 &= 449 \\
  C_7 &= 139 \\
  C_8 &= 383 \\
  C_9 &= 472 \\
  C_{10} &= 288 \\
  C_{11} &= 320 \\
  C_{12} &= 250 \\
  C_{13} &= 402 \\
  C_{14} &= 293 \\
  \end{align*}
  \]

- For each boat type $j \in B$, let $v_j$ be its value and $w_j$ its weight (size):

  \[
  \begin{align*}
  &\text{Speedboat:} \quad v = 69978, \quad w = 18 \\
  &\text{Fishing Boat:} \quad v = 54011, \quad w = 42 \\
  &\text{Catamaran:} \quad v = 36352, \quad w = 49 \\
  &\text{Yacht:} \quad v = 51521, \quad w = 42 \\
  &\text{Sailboat:} \quad v = 50415, \quad w = 41 \\
  &\text{Kayak:} \quad v = 76109, \quad w = 48 \\
  &\text{Canoe:} \quad v = 50462, \quad w = 22 \\
  &\text{Houseboat:} \quad v = 28989, \quad w = 29 \\
  &\text{Pontoon:} \quad v = 23318, \quad w = 45 \\
  &\text{Jet Ski:} \quad v = 26142, \quad w = 14 \\
  &\text{Rowboat:} \quad v = 42040, \quad w = 38 \\
  &\text{Hovercraft:} \quad v = 85961, \quad w = 47 \\
  &\text{Cabin Cruiser:} \quad v = 50142, \quad w = 45 \\
  &\text{Wakeboard Boat:} \quad v = 48478, \quad w = 28 \\
  &\text{Dinghy:} \quad v = 60953, \quad w = 24 \\
  &\text{Trawler:} \quad v = 95265, \quad w = 39 \\
  &\text{Paddle Boat:} \quad v = 22839, \quad w = 32 \\
  &\text{Submarine:} \quad v = 90957, \quad w = 36 \\
  &\text{RIB:} \quad v = 84652, \quad w = 14 \\
  &\text{Skiff:} \quad v = 78991, \quad w = 16 \\
  \end{align*}
  \]

---

### Mathematical Model

**Objective:**

\[
\max \sum_{i \in D} \sum_{j \in B} v_j \cdot x_{ij}
\]

**Subject to:**

For each display area $i \in D$:
\[
\sum_{j \in B} w_j \cdot x_{ij} \leq C_i
\]

For all $i \in D$, $j \in B$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Where:**

- $x_{ij}$ = number of units of boat type $j$ to place in display area $i$ (integer, $\geq 0$)
- $v_j$ = value of boat type $j$
- $w_j$ = weight (size) of boat type $j$
- $C_i$ = capacity of display area $i$

All identifiers and coefficients are as retrieved and shown above.