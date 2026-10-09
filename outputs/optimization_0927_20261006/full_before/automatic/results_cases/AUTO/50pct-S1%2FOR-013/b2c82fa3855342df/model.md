Let $x_{ij}$ be the number of units of air conditioner type $j$ (see ProductName) to be placed in storage area $i$ (see StorageID). All $x_{ij}$ are integer and $\geq 0$.

Maximize total value:
$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit},\,\text{Portable Unit},\,\text{Split System},\,\text{Ductless System},\,\text{Central AC},\,\text{Hybrid AC},\,\text{Geothermal AC},\,\text{Smart AC},\,\text{Evaporative Cooler},\,\text{Package Unit}\}} v_j \cdot x_{ij}
$$

where the values $v_j$ are:
- Window Unit: $4811$
- Portable Unit: $1130$
- Split System: $1611$
- Ductless System: $3368$
- Central AC: $2135$
- Hybrid AC: $1046$
- Geothermal AC: $4030$
- Smart AC: $3761$
- Evaporative Cooler: $3523$
- Package Unit: $1701$

Subject to, for each storage area $i$ (StorageID):

For StorageID $1$ (Capacity $1083$):
$$
114\,x_{1,\text{Window Unit}} + 200\,x_{1,\text{Portable Unit}} + 106\,x_{1,\text{Split System}} + 256\,x_{1,\text{Ductless System}} + 268\,x_{1,\text{Central AC}} + 185\,x_{1,\text{Hybrid AC}} + 299\,x_{1,\text{Geothermal AC}} + 131\,x_{1,\text{Smart AC}} + 139\,x_{1,\text{Evaporative Cooler}} + 105\,x_{1,\text{Package Unit}} \leq 1083
$$

For StorageID $2$ (Capacity $1840$):
$$
114\,x_{2,\text{Window Unit}} + 200\,x_{2,\text{Portable Unit}} + 106\,x_{2,\text{Split System}} + 256\,x_{2,\text{Ductless System}} + 268\,x_{2,\text{Central AC}} + 185\,x_{2,\text{Hybrid AC}} + 299\,x_{2,\text{Geothermal AC}} + 131\,x_{2,\text{Smart AC}} + 139\,x_{2,\text{Evaporative Cooler}} + 105\,x_{2,\text{Package Unit}} \leq 1840
$$

For StorageID $3$ (Capacity $770$):
$$
114\,x_{3,\text{Window Unit}} + 200\,x_{3,\text{Portable Unit}} + 106\,x_{3,\text{Split System}} + 256\,x_{3,\text{Ductless System}} + 268\,x_{3,\text{Central AC}} + 185\,x_{3,\text{Hybrid AC}} + 299\,x_{3,\text{Geothermal AC}} + 131\,x_{3,\text{Smart AC}} + 139\,x_{3,\text{Evaporative Cooler}} + 105\,x_{3,\text{Package Unit}} \leq 770
$$

For StorageID $4$ (Capacity $1299$):
$$
114\,x_{4,\text{Window Unit}} + 200\,x_{4,\text{Portable Unit}} + 106\,x_{4,\text{Split System}} + 256\,x_{4,\text{Ductless System}} + 268\,x_{4,\text{Central AC}} + 185\,x_{4,\text{Hybrid AC}} + 299\,x_{4,\text{Geothermal AC}} + 131\,x_{4,\text{Smart AC}} + 139\,x_{4,\text{Evaporative Cooler}} + 105\,x_{4,\text{Package Unit}} \leq 1299
$$

For StorageID $5$ (Capacity $1259$):
$$
114\,x_{5,\text{Window Unit}} + 200\,x_{5,\text{Portable Unit}} + 106\,x_{5,\text{Split System}} + 256\,x_{5,\text{Ductless System}} + 268\,x_{5,\text{Central AC}} + 185\,x_{5,\text{Hybrid AC}} + 299\,x_{5,\text{Geothermal AC}} + 131\,x_{5,\text{Smart AC}} + 139\,x_{5,\text{Evaporative Cooler}} + 105\,x_{5,\text{Package Unit}} \leq 1259
$$

For StorageID $6$ (Capacity $543$):
$$
114\,x_{6,\text{Window Unit}} + 200\,x_{6,\text{Portable Unit}} + 106\,x_{6,\text{Split System}} + 256\,x_{6,\text{Ductless System}} + 268\,x_{6,\text{Central AC}} + 185\,x_{6,\text{Hybrid AC}} + 299\,x_{6,\text{Geothermal AC}} + 131\,x_{6,\text{Smart AC}} + 139\,x_{6,\text{Evaporative Cooler}} + 105\,x_{6,\text{Package Unit}} \leq 543
$$

For StorageID $7$ (Capacity $1831$):
$$
114\,x_{7,\text{Window Unit}} + 200\,x_{7,\text{Portable Unit}} + 106\,x_{7,\text{Split System}} + 256\,x_{7,\text{Ductless System}} + 268\,x_{7,\text{Central AC}} + 185\,x_{7,\text{Hybrid AC}} + 299\,x_{7,\text{Geothermal AC}} + 131\,x_{7,\text{Smart AC}} + 139\,x_{7,\text{Evaporative Cooler}} + 105\,x_{7,\text{Package Unit}} \leq 1831
$$

For StorageID $8$ (Capacity $855$):
$$
114\,x_{8,\text{Window Unit}} + 200\,x_{8,\text{Portable Unit}} + 106\,x_{8,\text{Split System}} + 256\,x_{8,\text{Ductless System}} + 268\,x_{8,\text{Central AC}} + 185\,x_{8,\text{Hybrid AC}} + 299\,x_{8,\text{Geothermal AC}} + 131\,x_{8,\text{Smart AC}} + 139\,x_{8,\text{Evaporative Cooler}} + 105\,x_{8,\text{Package Unit}} \leq 855
$$

For StorageID $9$ (Capacity $619$):
$$
114\,x_{9,\text{Window Unit}} + 200\,x_{9,\text{Portable Unit}} + 106\,x_{9,\text{Split System}} + 256\,x_{9,\text{Ductless System}} + 268\,x_{9,\text{Central AC}} + 185\,x_{9,\text{Hybrid AC}} + 299\,x_{9,\text{Geothermal AC}} + 131\,x_{9,\text{Smart AC}} + 139\,x_{9,\text{Evaporative Cooler}} + 105\,x_{9,\text{Package Unit}} \leq 619
$$

For StorageID $10$ (Capacity $637$):
$$
114\,x_{10,\text{Window Unit}} + 200\,x_{10,\text{Portable Unit}} + 106\,x_{10,\text{Split System}} + 256\,x_{10,\text{Ductless System}} + 268\,x_{10,\text{Central AC}} + 185\,x_{10,\text{Hybrid AC}} + 299\,x_{10,\text{Geothermal AC}} + 131\,x_{10,\text{Smart AC}} + 139\,x_{10,\text{Evaporative Cooler}} + 105\,x_{10,\text{Package Unit}} \leq 637
$$

For StorageID $11$ (Capacity $935$):
$$
114\,x_{11,\text{Window Unit}} + 200\,x_{11,\text{Portable Unit}} + 106\,x_{11,\text{Split System}} + 256\,x_{11,\text{Ductless System}} + 268\,x_{11,\text{Central AC}} + 185\,x_{11,\text{Hybrid AC}} + 299\,x_{11,\text{Geothermal AC}} + 131\,x_{11,\text{Smart AC}} + 139\,x_{11,\text{Evaporative Cooler}} + 105\,x_{11,\text{Package Unit}} \leq 935
$$

For StorageID $12$ (Capacity $626$):
$$
114\,x_{12,\text{Window Unit}} + 200\,x_{12,\text{Portable Unit}} + 106\,x_{12,\text{Split System}} + 256\,x_{12,\text{Ductless System}} + 268\,x_{12,\text{Central AC}} + 185\,x_{12,\text{Hybrid AC}} + 299\,x_{12,\text{Geothermal AC}} + 131\,x_{12,\text{Smart AC}} + 139\,x_{12,\text{Evaporative Cooler}} + 105\,x_{12,\text{Package Unit}} \leq 626
$$

For StorageID $13$ (Capacity $1457$):
$$
114\,x_{13,\text{Window Unit}} + 200\,x_{13,\text{Portable Unit}} + 106\,x_{13,\text{Split System}} + 256\,x_{13,\text{Ductless System}} + 268\,x_{13,\text{Central AC}} + 185\,x_{13,\text{Hybrid AC}} + 299\,x_{13,\text{Geothermal AC}} + 131\,x_{13,\text{Smart AC}} + 139\,x_{13,\text{Evaporative Cooler}} + 105\,x_{13,\text{Package Unit}} \leq 1457
$$

For StorageID $14$ (Capacity $1198$):
$$
114\,x_{14,\text{Window Unit}} + 200\,x_{14,\text{Portable Unit}} + 106\,x_{14,\text{Split System}} + 256\,x_{14,\text{Ductless System}} + 268\,x_{14,\text{Central AC}} + 185\,x_{14,\text{Hybrid AC}} + 299\,x_{14,\text{Geothermal AC}} + 131\,x_{14,\text{Smart AC}} + 139\,x_{14,\text{Evaporative Cooler}} + 105\,x_{14,\text{Package Unit}} \leq 1198
$$

For StorageID $15$ (Capacity $837$):
$$
114\,x_{15,\text{Window Unit}} + 200\,x_{15,\text{Portable Unit}} + 106\,x_{15,\text{Split System}} + 256\,x_{15,\text{Ductless System}} + 268\,x_{15,\text{Central AC}} + 185\,x_{15,\text{Hybrid AC}} + 299\,x_{15,\text{Geothermal AC}} + 131\,x_{15,\text{Smart AC}} + 139\,x_{15,\text{Evaporative Cooler}} + 105\,x_{15,\text{Package Unit}} \leq 837
$$

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,15\},\ j \in \{\text{Window Unit},\,\text{Portable Unit},\,\text{Split System},\,\text{Ductless System},\,\text{Central AC},\,\text{Hybrid AC},\,\text{Geothermal AC},\,\text{Smart AC},\,\text{Evaporative Cooler},\,\text{Package Unit}\}
$$

All coefficients and identifiers are as retrieved and in original order.