Let $x_{ij}$ be the number of vessels of type $j$ (ProductName $j$) to be placed in display area $i$ (DisplayID $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $V_j$: Value of vessel type $j$
- $W_j$: Weight (size) of vessel type $j$
- $C_i$: Capacity of display area $i$

**Sets:**

- $i \in \{1,2,\ldots,14\}$ (DisplayID from capacity.csv)
- $j \in \{$Speedboat, Fishing Boat, Catamaran, Yacht, Sailboat, Kayak, Canoe, Houseboat, Pontoon, Jet Ski, Rowboat, Hovercraft, Cabin Cruiser, Wakeboard Boat, Dinghy, Trawler, Paddle Boat, Submarine, RIB, Skiff$\}$ (ProductName from products.csv)

---

**Objective:**

$$
\max \sum_{i=1}^{14} \sum_{j} V_j \cdot x_{ij}
$$

where

- $V_j$ (Value) for each $j$:

  - Speedboat: 29664
  - Fishing Boat: 31778
  - Catamaran: 73501
  - Yacht: 78255
  - Sailboat: 93606
  - Kayak: 46983
  - Canoe: 95026
  - Houseboat: 57685
  - Pontoon: 60323
  - Jet Ski: 91224
  - Rowboat: 44003
  - Hovercraft: 75998
  - Cabin Cruiser: 84525
  - Wakeboard Boat: 66207
  - Dinghy: 65002
  - Trawler: 33132
  - Paddle Boat: 69239
  - Submarine: 66948
  - RIB: 88240
  - Skiff: 48858

---

**Constraints:**

For each display area $i$ (DisplayID):

$$
\sum_{j} W_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,14
$$

where

- $W_j$ (Weight) for each $j$:

  - Speedboat: 18
  - Fishing Boat: 36
  - Catamaran: 25
  - Yacht: 16
  - Sailboat: 97
  - Kayak: 35
  - Canoe: 32
  - Houseboat: 100
  - Pontoon: 43
  - Jet Ski: 15
  - Rowboat: 95
  - Hovercraft: 57
  - Cabin Cruiser: 13
  - Wakeboard Boat: 44
  - Dinghy: 64
  - Trawler: 88
  - Paddle Boat: 42
  - Submarine: 46
  - RIB: 24
  - Skiff: 93

- $C_i$ (Capacity) for each $i$:

  - 1: 457
  - 2: 604
  - 3: 751
  - 4: 468
  - 5: 343
  - 6: 408
  - 7: 741
  - 8: 914
  - 9: 682
  - 10: 409
  - 11: 342
  - 12: 903
  - 13: 680
  - 14: 886

---

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i=1}^{14} \sum_{j} V_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j} W_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,14 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\end{align*}
$$

---

**Retrieved Data Used:**

- Display areas (DisplayID) and their capacities (Capacity) from capacity.csv, in source order.
- Vessel types (ProductName), their values (Value), and sizes (Weight) from products.csv, in source order.
- All variables, coefficients, and constraints are indexed by the explicit identifiers from the data.