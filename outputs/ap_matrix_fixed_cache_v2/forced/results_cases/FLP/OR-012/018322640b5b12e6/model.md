##### Decision Variables

Let $x_i \geq 0$ denote the number of units of product $i$ to fulfill for customer purchases, for each product $i$ in the set of all products.

##### Parameters

Let $P$ be the set of all products, indexed by $i$.

For each product $i \in P$:
- $r_i$: revenue per unit of product $i$ (from the 'Revenue' column)
- $d_i$: demand for product $i$ (from the 'Demand' column)
- $s_i$: initial inventory of product $i$ (from the 'Initial Inventory' column)

The data is as follows (partial list for illustration; full data is included in the model):

| Product Name                                      | $r_i$   | $d_i$ | $s_i$ |
|---------------------------------------------------|---------|-------|-------|
| 1984 by George Orwell                             | 19.99   | 6     | 20    |
| Adidas 3-Stripes Shorts                           | 29.99   | 7     | 30    |
| Adidas Essential Track Pants                      | 44.99   | 5     | 20    |
| Adidas FIFA World Cup Football                    | 29.99   | 4     | 20    |
| Adidas Originals Superstar Sneakers               | 79.99   | 6     | 20    |
| Adidas Originals Trefoil Hoodie                   | 64.99   | 6     | 20    |
| Adidas Ultraboost Running Shoes                   | 179.99  | 3     | 10    |
| Adidas Ultraboost Shoes                           | 179.99  | 3     | 10    |
| Amazon Echo Dot (4th Gen)                         | 49.99   | 5     | 20    |
| Amazon Echo Show 10                               | 249.99  | 2     | 10    |
| ...                                               | ...     | ...   | ...   |

(Full data: see below.)

##### Objective Function

\[
\max \sum_{i \in P} r_i x_i
\]

##### Constraints

1. **Demand and Inventory Fulfillment:**
   \[
   0 \leq x_i \leq \min\{d_i, s_i\}, \quad \forall i \in P
   \]
   That is, for each product, the number of units fulfilled cannot exceed either the available inventory or the demand.

##### Variable Domains

\[
x_i \geq 0 \quad \text{and integer (if only whole units can be sold)}, \quad \forall i \in P
\]

##### Full Parameter Data

Let $P$ be the set of all products with the following parameters:

- 1984 by George Orwell: $r_i = 19.99$, $d_i = 6$, $s_i = 20$
- Adidas 3-Stripes Shorts: $r_i = 29.99$, $d_i = 7$, $s_i = 30$
- Adidas Essential Track Pants: $r_i = 44.99$, $d_i = 5$, $s_i = 20$
- Adidas FIFA World Cup Football: $r_i = 29.99$, $d_i = 4$, $s_i = 20$
- Adidas Originals Superstar Sneakers: $r_i = 79.99$, $d_i = 6$, $s_i = 20$
- Adidas Originals Trefoil Hoodie: $r_i = 64.99$, $d_i = 6$, $s_i = 20$
- Adidas Ultraboost Running Shoes: $r_i = 179.99$, $d_i = 3$, $s_i = 10$
- Adidas Ultraboost Shoes: $r_i = 179.99$, $d_i = 3$, $s_i = 10$
- Amazon Echo Dot (4th Gen): $r_i = 49.99$, $d_i = 5$, $s_i = 20$
- Amazon Echo Show 10: $r_i = 249.99$, $d_i = 2$, $s_i = 10$
- Amazon Fire TV Stick 4K: $r_i = 49.99$, $d_i = 5$, $s_i = 20$
- Anastasia Beverly Hills Brow Wiz: $r_i = 23$, $d_i = 3$, $s_i = 10$
- Anker PowerCore Portable Charger: $r_i = 59.99$, $d_i = 6$, $s_i = 20$
- Anova Precision Cooker: $r_i = 199$, $d_i = 3$, $s_i = 10$
- Anova Precision Oven: $r_i = 599$, $d_i = 2$, $s_i = 10$
- Apple AirPods Max: $r_i = 549$, $d_i = 2$, $s_i = 10$
- Apple AirPods Pro: $r_i = 249.99$, $d_i = 3$, $s_i = 10$
- Apple MacBook Air: $r_i = 1199.99$, $d_i = 2$, $s_i = 10$
- Apple MacBook Pro 16-inch: $r_i = 2399$, $d_i = 2$, $s_i = 10$
- Apple TV 4K: $r_i = 179$, $d_i = 3$, $s_i = 10$
- Apple Watch Series 8: $r_i = 399.99$, $d_i = 4$, $s_i = 20$
- Apple iPad Air: $r_i = 599.99$, $d_i = 3$, $s_i = 10$
- Atomic Habits by James Clear: $r_i = 16.99$, $d_i = 6$, $s_i = 20$
- Babolat Pure Drive Tennis Racket: $r_i = 199.99$, $d_i = 5$, $s_i = 20$
- Becoming by Michelle Obama: $r_i = 32.5$, $d_i = 6$, $s_i = 20$
- Biore UV Aqua Rich Watery Essence Sunscreen: $r_i = 15$, $d_i = 2$, $s_i = 10$
- Blueair Classic 480i: $r_i = 599.99$, $d_i = 3$, $s_i = 10$
- Bose QuietComfort 35 Headphones: $r_i = 299.99$, $d_i = 2$, $s_i = 10$
- Bose QuietComfort 35 II Wireless Headphones: $r_i = 299$, $d_i = 2$, $s_i = 10$
- Bose SoundLink Color Bluetooth Speaker II: $r_i = 129$, $d_i = 2$, $s_i = 10$
- Bose SoundLink Revolve+ Speaker: $r_i = 299.99$, $d_i = 5$, $s_i = 20$
- Bose SoundSport Wireless Earbuds: $r_i = 149.99$, $d_i = 3$, $s_i = 10$
- Bowflex SelectTech 1090 Adjustable Dumbbells: $r_i = 699.99$, $d_i = 2$, $s_i = 10$
- Bowflex SelectTech 552 Dumbbells: $r_i = 399.99$, $d_i = 2$, $s_i = 10$
- Breville Nespresso Creatista Plus: $r_i = 499.95$, $d_i = 2$, $s_i = 10$
- Breville Smart Coffee Grinder Pro: $r_i = 199.95$, $d_i = 2$, $s_i = 10$
- Breville Smart Grill: $r_i = 299.95$, $d_i = 3$, $s_i = 10$
- Breville Smart Oven: $r_i = 299.99$, $d_i = 2$, $s_i = 10$
- Calvin Klein Boxer Briefs: $r_i = 29.99$, $d_i = 7$, $s_i = 30$
- Canon EOS R5 Camera: $r_i = 3899.99$, $d_i = 2$, $s_i = 10$
- Canon EOS Rebel T7i DSLR Camera: $r_i = 749.99$, $d_i = 2$, $s_i = 10$
- Caudalie Vinoperfect Radiance Serum: $r_i = 79$, $d_i = 2$, $s_i = 10$
- CeraVe Hydrating Facial Cleanser: $r_i = 14.99$, $d_i = 3$, $s_i = 10$
- Champion Reverse Weave Hoodie: $r_i = 49.99$, $d_i = 5$, $s_i = 20$
- Chanel No. 5 Perfume: $r_i = 129.99$, $d_i = 2$, $s_i = 10$
- Charlotte Tilbury Magic Cream: $r_i = 100$, $d_i = 2$, $s_i = 10$
- Clinique Dramatically Different Moisturizing Lotion: $r_i = 29.5$, $d_i = 2$, $s_i = 10$
- Clinique Moisture Surge: $r_i = 52$, $d_i = 2$, $s_i = 10$
- Columbia Fleece Jacket: $r_i = 59.99$, $d_i = 6$, $s_i = 20$
- Crock-Pot 6-Quart Slow Cooker: $r_i = 49.99$, $d_i = 3$, $s_i = 10$
- Cuisinart Coffee Center: $r_i = 199.95$, $d_i = 3$, $s_i = 10$
- Cuisinart Custom 14-Cup Food Processor: $r_i = 199.99$, $d_i = 2$, $s_i = 10$
- Cuisinart Griddler Deluxe: $r_i = 159.99$, $d_i = 2$, $s_i = 10$
- De'Longhi Magnifica Espresso Machine: $r_i = 899.99$, $d_i = 2$, $s_i = 10$
- Dr. Jart+ Cicapair Tiger Grass Color Correcting Treatment: $r_i = 52$, $d_i = 2$, $s_i = 10$
- Drunk Elephant C-Firma Day Serum: $r_i = 78$, $d_i = 2$, $s_i = 10$
- Dune by Frank Herbert: $r_i = 25.99$, $d_i = 6$, $s_i = 20$
- Dyson Pure Cool Link: $r_i = 499.99$, $d_i = 2$, $s_i = 10$
- Dyson Supersonic Hair Dryer: $r_i = 399.99$, $d_i = 5$, $s_i = 20$
- Dyson V11 Vacuum: $r_i = 499.99$, $d_i = 2$, $s_i = 10$
- Dyson V8 Absolute: $r_i = 399.99$, $d_i = 2$, $s_i = 10$
- Educated by Tara Westover: $r_i = 28$, $d_i = 4$, $s_i = 20$
- Estee Lauder Advanced Night Repair: $r_i = 105$, $d_i = 2$, $s_i = 10$
- Eufy RoboVac 11S: $r_i = 219.99$, $d_i = 5$, $s_i = 20$
- Fenty Beauty Killawatt Highlighter: $r_i = 36$, $d_i = 2$, $s_i = 10$
- First Aid Beauty Ultra Repair Cream: $r_i = 34$, $d_i = 3$, $s_i = 10$
- Fitbit Charge 5: $r_i = 129.99$, $d_i = 3$, $s_i = 10$
- Fitbit Inspire 2: $r_i = 99.95$, $d_i = 3$, $s_i = 10$
- Fitbit Luxe: $r_i = 149.95$, $d_i = 3$, $s_i = 10$
- Fitbit Versa 3: $r_i = 229.95$, $d_i = 5$, $s_i = 20$
- Forever 21 Graphic Tee: $r_i = 12.99$, $d_i = 7$, $s_i = 30$
- Fresh Sugar Lip Treatment: $r_i = 24$, $d_i = 2$, $s_i = 10$
- Gap 1969 Original Fit Jeans: $r_i = 59.99$, $d_i = 5$, $s_i = 20$
- Gap Crewneck Sweatshirt: $r_i = 34.99$, $d_i = 6$, $s_i = 20$
- Gap Essential Crewneck T-Shirt: $r_i = 19.99$, $d_i = 8$, $s_i = 30$
- Gap High Rise Skinny Jeans: $r_i = 49.99$, $d_i = 5$, $s_i = 20$
- Garmin Edge 530: $r_i = 299.99$, $d_i = 3$, $s_i = 10$
- Garmin Fenix 6X Pro: $r_i = 999.99$, $d_i = 2$, $s_i = 10$
- Garmin Forerunner 245: $r_i = 299.99$, $d_i = 2$, $s_i = 10$
- Garmin Forerunner 945: $r_i = 499.99$, $d_i = 5$, $s_i = 20$
- GlamGlow Supermud Clearing Treatment: $r_i = 59$, $d_i = 2$, $s_i = 10$
- Glossier Boy Brow: $r_i = 16$, $d_i = 3$, $s_i = 10$
- Glossier Cloud Paint: $r_i = 18$, $d_i = 2$, $s_i = 10$
- GoPro HERO10 Black: $r_i = 399.99$, $d_i = 4$, $s_i = 20$
- GoPro HERO9 Black: $r_i = 449.99$, $d_i = 2$, $s_i = 10$
- Gone Girl by Gillian Flynn: $r_i = 22.99$, $d_i = 3$, $s_i = 10$
- Google Nest Hub Max: $r_i = 229.99$, $d_i = 3$, $s_i = 10$
- Google Nest Wifi Router: $r_i = 169$, $d_i = 2$, $s_i = 10$
- Google Pixel 6 Pro: $r_i = 899.99$, $d_i = 2$, $s_i = 10$
- Google Pixelbook Go: $r_i = 649.99$, $d_i = 2$, $s_i = 10$
- H&M Slim Fit Jeans: $r_i = 39.99$, $d_i = 5$, $s_i = 20$
- HP Spectre x360 Laptop: $r_i = 1599.99$, $d_i = 2$, $s_i = 10$
- Hamilton Beach FlexBrew Coffee Maker: $r_i = 89.99$, $d_i = 2$, $s_i = 10$
- Hanes ComfortSoft T-Shirt: $r_i = 9.99$, $d_i = 15$, $s_i = 50$
- Harry Potter and the Sorcerer's Stone: $r_i = 24.99$, $d_i = 4$, $s_i = 20$
- Hydro Flask Standard Mouth Water Bottle: $r_i = 32.95$, $d_i = 5$, $s_i = 20$
- Hydro Flask Wide Mouth Water Bottle: $r_i = 39.95$, $d_i = 6$, $s_i = 20$
- Hyperice Hypervolt Massager: $r_i = 349$, $d_i = 2$, $s_i = 10$
- Instant Pot Duo: $r_i = 89.99$, $d_i = 4$, $s_i = 20$
- Instant Pot Duo Crisp: $r_i = 179.99$, $d_i = 2$, $s_i = 10$
- Instant Pot Duo Evo Plus: $r_i = 139.99$, $d_i = 3$, $s_i = 10$
- Instant Pot Duo Nova: $r_i = 99.95$, $d_i = 2$, $s_i = 10$
- Instant Pot Ultra: $r_i = 139.99$, $d_i = 2$, $s_i = 10$
- Keurig K-Elite Coffee Maker: $r_i = 189.99$, $d_i = 5$, $s_i = 20$
- Keurig K-Mini Coffee Maker: $r_i = 79.99$, $d_i = 3$, $s_i = 10$
- Kiehl's Midnight Recovery Concentrate: $r_i = 82$, $d_i = 2$, $s_i = 10$
- Kindle Paperwhite: $r_i = 129.99$, $d_i = 3$, $s_i = 10$
- KitchenAid Artisan Stand Mixer: $r_i = 499.99$, $d_i = 2$, $s_i = 10$
- KitchenAid Stand Mixer: $r_i = 379.99$, $d_i = 2$, $s_i = 10$
- L'Occitane Shea Butter Hand Cream: $r_i = 29$, $d_i = 3$, $s_i = 10$
- L'Oreal Revitalift Serum: $r_i = 39.99$, $d_i = 3$, $s_i = 10$
- LG OLED TV: $r_i = 1299.99$, $d_i = 3$, $s_i = 10$
- La Mer Cr_¨me de la Mer Moisturizer: $r_i = 190$, $d_i = 2$, $s_i = 10$
- Lancome La Vie Est Belle: $r_i = 102$, $d_i = 2$, $s_i = 10$
- Laneige Water Sleeping Mask: $r_i = 25$, $d_i = 2$, $s_i = 10$
- Levi's 501 Jeans: $r_i = 69.99$, $d_i = 5$, $s_i = 20$
- Levi's 511 Slim Fit Jeans: $r_i = 59.99$, $d_i = 5$, $s_i = 20$
- Levi's Sherpa Trucker Jacket: $r_i = 98$, $d_i = 3$, $s_i = 10$
- Levi's Trucker Jacket: $r_i = 89.99$, $d_i = 3$, $s_i = 10$

(And so on for all products in the dataset.)

##### Summary

- Decision variables: $x_i$ (units fulfilled for each product $i$)
- Objective: maximize total revenue
- Constraints: $0 \leq x_i \leq \min\{d_i, s_i\}$ for all $i$
- All parameters ($r_i$, $d_i$, $s_i$) are as listed above for each product.

This model ensures that for each product, the retailer fulfills as many units as possible up to the lesser of demand and available inventory, maximizing total revenue.