Let $i$ index the products as given by the "Product Name" column (preserving source order). Let $x_i$ be the number of units of product $i$ to fulfill.

Objective:
$$
\max \sum_{i} r_i x_i
$$
where $r_i$ is the "Revenue" for product $i$.

Subject to, for each product $i$:
$$
0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\}
$$
$$
x_i \in \mathbb{Z}_{\geq 0}
$$

where:
- $\text{Demand}_i$ is the "Demand" for product $i$
- $\text{Initial Inventory}_i$ is the "Initial Inventory" for product $i$

Explicitly, for each product $i$ (in source order):

| $i$ | Product Name | $r_i$ | Demand$_i$ | Initial Inventory$_i$ |
|-----|-------------|-------|------------|-----------------------|
| 1 |  3 Colors New Fashion Summer Ladies Casual Jumpsuit Long Suspender Overalls Bib Pants | 11 | 1357 | 5000 |
| 2 | Men's Casual Workout Athletic Gym Jersey Shorts Elastic Waist Drawstring Summer Training Running Knee Length Shorts with Zipper Pocket | 8 | 1363 | 5000 |
| 3 | New Fashion Autumn Summer Women's Long Sleeve V Neck Long Dress Floral Print Split Maxi Dress Holiday Party Beach Sundress Evening Dresses | 19 | 1269 | 5000 |
| 4 | New The New Men's Stitching Design Jogging Sports Cropped Trousers | 9 | 6500 | 25000 |
| 5 | Plus Size S-5XL Women Summer Tops Casual V-Neck Short Sleeve Shirts Ladies Cotton Loose T Shirt Candy Color Lady Pullovers Blouse | 5 | 6335 | 25000 |
| 6 | Plus Size Women Halter Striped Wide Leg Pants Casual Jumpsuit Rompers Fashion Shorts  | 8 | 1283 | 5000 |
| 7 | Spring/Summer Fashion Women "honey"Letter Print Sleeveless Shirt Sexy Embroidered Bodycon Vest Knitted Cotton Vest Casual Tank Top  | 5.65 | 27327 | 100000 |
| 8 | Summer Students Style Bodycon Rompers Womens Slim Fit Jumpsuit Chest Zipper Contrast Color Short Sleeve V-neck Shorts Bodysuit | 11 | 25600 | 100000 |
| 9 | Summer Women s Fashion Lace Up Tie Pants Plus Size Casual High Waist Short Pants(S-5XL) | 4.93 | 264 | 1000 |
| 10 | Women's Fashion Graphic \Don't Flatter Yourself....\Tee for Women Summer Casual Tee T Shirts for Girls | 7 | 14 | 50 |
| ... | ... | ... | ... | ... |
| 75 | 2019 New Fashion Women's Slim Plus Size Maxi Dress Ink Printing Spaghetti Strap Dress | 11 | 28189 | 100000 |

Decision variables:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad 0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\}, \quad \forall i=1,\ldots,75
$$

where all coefficients and bounds are as given in the table above (full data preserved in source order).