Let $x_{ij}$ be the number of units of boat type $j$ (ProductName) to be placed in display area $i$ (DisplayID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $D$ be the set of display areas (DisplayID):  
  $D = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14\}$

- Let $B$ be the set of boat types (ProductName):  
  $B = \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$

- For each display area $i \in D$, let $C_i$ be its capacity:
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

- For each boat type $j \in B$, let $v_j$ be its value and $w_j$ its weight (size):

  | ProductName        | $v_j$  | $w_j$ |
  |--------------------|--------|-------|
  | Speedboat          | 69978  | 18    |
  | Fishing Boat       | 54011  | 42    |
  | Catamaran          | 36352  | 49    |
  | Yacht              | 51521  | 42    |
  | Sailboat           | 50415  | 41    |
  | Kayak              | 76109  | 48    |
  | Canoe              | 50462  | 22    |
  | Houseboat          | 28989  | 29    |
  | Pontoon            | 23318  | 45    |
  | Jet Ski            | 26142  | 14    |
  | Rowboat            | 42040  | 38    |
  | Hovercraft         | 85961  | 47    |
  | Cabin Cruiser      | 50142  | 45    |
  | Wakeboard Boat     | 48478  | 28    |
  | Dinghy             | 60953  | 24    |
  | Trawler            | 95265  | 39    |
  | Paddle Boat        | 22839  | 32    |
  | Submarine          | 90957  | 36    |
  | RIB                | 84652  | 14    |
  | Skiff              | 78991  | 16    |

---

**Mathematical Model**

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in D, \forall j \in B
$$

**Objective:**
$$
\max \sum_{i \in D} \sum_{j \in B} v_j \cdot x_{ij}
$$

**Subject to:**

For each display area $i \in D$:
$$
\sum_{j \in B} w_j \cdot x_{ij} \leq C_i
$$

For all $i \in D$, $j \in B$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**All identifiers and coefficients:**

- Display areas (DisplayID): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
- Boat types (ProductName): Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff
- Values ($v_j$): as listed above
- Weights ($w_j$): as listed above
- Capacities ($C_i$): as listed above

**Variable domains:**  
$x_{ij}$ are nonnegative integers for all $i, j$.