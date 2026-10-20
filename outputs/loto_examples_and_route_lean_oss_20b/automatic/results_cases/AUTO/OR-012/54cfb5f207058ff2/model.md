##### Objective Function: 

$\quad \quad \max \sum_{i} \text{Revenue}_{i} \cdot x_{i}$

##### Constraints

###### 1. Demand Constraints: 

$\quad \quad x_{i} \leq \text{Demand}_{i}, \quad \forall i$

###### 2. Inventory Constraints: 

$\quad \quad x_{i} \leq \text{Initial Inventory}_{i}, \quad \forall i$

###### 3. Non-negativity and Integrality Constraints: 

$\quad \quad x_{i} \in \mathbb{Z}_{\geq 0}, \quad \forall i$

###### 4. Variable Definition: 

$x_{i}$ denotes the number of units of product $i$ that the retailer plans to fulfill for customer purchases.

###### 5. Variable Domain: 

$x_{i}$ are nonnegative integers.

###### 6. Objective Sense: 

Maximize total revenue.

###### 7. Data Mapping: 

The following JSON object lists all products in the order they appear in the dataset, along with their revenue, demand, and initial inventory values.

###### Retrieved Information

```json
{
  "products": [
    { "name": "1984 by George Orwell", "revenue": 19.99, "demand": 6, "initial_inventory": 20 },
    { "name": "Adidas 3-Stripes Shorts", "revenue": 29.99, "demand": 7, "initial_inventory": 30 },
    { "name": "Adidas Essential Track Pants", "revenue": 44.99, "demand": 5, "initial_inventory": 20 },
    { "name": "Adidas FIFA World Cup Football", "revenue": 29.99, "demand": 4, "initial_inventory": 20 },
    { "name": "Adidas Originals Superstar Sneakers", "revenue": 79.99, "demand": 6, "initial_inventory": 20 },
    { "name": "Adidas Originals Trefoil Hoodie", "revenue": 64.99, "demand": 6, "initial_inventory": 20 },
    { "name": "Adidas Ultraboost Running Shoes", "revenue": 179.99, "demand": 3, "initial_inventory": 10 },
    { "name": "Adidas Ultraboost Shoes", "revenue": 179.99, "demand": 3, "initial_inventory": 10 },
    { "name": "Amazon Echo Dot (4th Gen)", "revenue": 49.99, "demand": 5, "initial_inventory": 20 },
    { "name": "Amazon Echo Show 10", "revenue": 249.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Amazon Fire TV Stick 4K", "revenue": 49.99, "demand": 5, "initial_inventory": 20 },
    { "name": "Anastasia Beverly Hills Brow Wiz", "revenue": 23, "demand": 3, "initial_inventory": 10 },
    { "name": "Anker PowerCore Portable Charger", "revenue": 59.99, "demand": 6, "initial_inventory": 20 },
    { "name": "Anova Precision Cooker", "revenue": 199, "demand": 3, "initial_inventory": 10 },
    { "name": "Anova Precision Oven", "revenue": 599, "demand": 2, "initial_inventory": 10 },
    { "name": "Apple AirPods Max", "revenue": 549, "demand": 2, "initial_inventory": 10 },
    { "name": "Apple AirPods Pro", "revenue": 249.99, "demand": 3, "initial_inventory": 10 },
    { "name": "Apple MacBook Air", "revenue": 1199.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Apple MacBook Pro 16-inch", "revenue": 2399, "demand": 2, "initial_inventory": 10 },
    { "name": "Apple TV 4K", "revenue": 179, "demand": 3, "initial_inventory": 10 },
    { "name": "Apple Watch Series 8", "revenue": 399.99, "demand": 4, "initial_inventory": 20 },
    { "name": "Apple iPad Air", "revenue": 599.99, "demand": 3, "initial_inventory": 10 },
    { "name": "Atomic Habits by James Clear", "revenue": 16.99, "demand": 6, "initial_inventory": 20 },
    { "name": "Babolat Pure Drive Tennis Racket", "revenue": 199.99, "demand": 5, "initial_inventory": 20 },
    { "name": "Becoming by Michelle Obama", "revenue": 32.5, "demand": 6, "initial_inventory": 20 },
    { "name": "Biore UV Aqua Rich Watery Essence Sunscreen", "revenue": 15, "demand": 2, "initial_inventory": 10 },
    { "name": "Blueair Classic 480i", "revenue": 599.99, "demand": 3, "initial_inventory": 10 },
    { "name": "Bose QuietComfort 35 Headphones", "revenue": 299.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Bose QuietComfort 35 II Wireless Headphones", "revenue": 299, "demand": 2, "initial_inventory": 10 },
    { "name": "Bose SoundLink Color Bluetooth Speaker II", "revenue": 129, "demand": 2, "initial_inventory": 10 },
    { "name": "Bose SoundLink Revolve+ Speaker", "revenue": 299.99, "demand": 5, "initial_inventory": 20 },
    { "name": "Bose SoundSport Wireless Earbuds", "revenue": 149.99, "demand": 3, "initial_inventory": 10 },
    { "name": "Bowflex SelectTech 1090 Adjustable Dumbbells", "revenue": 699.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Bowflex SelectTech 552 Dumbbells", "revenue": 399.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Breville Nespresso Creatista Plus", "revenue": 499.95, "demand": 2, "initial_inventory": 10 },
    { "name": "Breville Smart Coffee Grinder Pro", "revenue": 199.95, "demand": 2, "initial_inventory": 10 },
    { "name": "Breville Smart Grill", "revenue": 299.95, "demand": 3, "initial_inventory": 10 },
    { "name": "Breville Smart Oven", "revenue": 299.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Calvin Klein Boxer Briefs", "revenue": 29.99, "demand": 7, "initial_inventory": 30 },
    { "name": "Canon EOS R5 Camera", "revenue": 3899.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Canon EOS Rebel T7i DSLR Camera", "revenue": 749.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Caudalie Vinoperfect Radiance Serum", "revenue": 79, "demand": 2, "initial_inventory": 10 },
    { "name": "CeraVe Hydrating Facial Cleanser", "revenue": 14.99, "demand": 3, "initial_inventory": 10 },
    { "name": "Champion Reverse Weave Hoodie", "revenue": 49.99, "demand": 5, "initial_inventory": 20 },
    { "name": "Chanel No. 5 Perfume", "revenue": 129.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Charlotte Tilbury Magic Cream", "revenue": 100, "demand": 2, "initial_inventory": 10 },
    { "name": "Clinique Dramatically Different Moisturizing Lotion", "revenue": 29.5, "demand": 2, "initial_inventory": 10 },
    { "name": "Clinique Moisture Surge", "revenue": 52, "demand": 2, "initial_inventory": 10 },
    { "name": "Columbia Fleece Jacket", "revenue": 59.99, "demand": 6, "initial_inventory": 20 },
    { "name": "Crock-Pot 6-Quart Slow Cooker", "revenue": 49.99, "demand": 3, "initial_inventory": 10 },
    { "name": "Cuisinart Coffee Center", "revenue": 199.95, "demand": 3, "initial_inventory": 10 },
    { "name": "Cuisinart Custom 14-Cup Food Processor", "revenue": 199.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Cuisinart Griddler Deluxe", "revenue": 159.99, "demand": 2, "initial_inventory": 10 },
    { "name": "De'Longhi Magnifica Espresso Machine", "revenue": 899.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Dr. Jart+ Cicapair Tiger Grass Color Correcting Treatment", "revenue": 52, "demand": 2, "initial_inventory": 10 },
    { "name": "Drunk Elephant C-Firma Day Serum", "revenue": 78, "demand": 2, "initial_inventory": 10 },
    { "name": "Dune by Frank Herbert", "revenue": 25.99, "demand": 6, "initial_inventory": 20 },
    { "name": "Dyson Pure Cool Link", "revenue": 499.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Dyson Supersonic Hair Dryer", "revenue": 399.99, "demand": 5, "initial_inventory": 20 },
    { "name": "Dyson V11 Vacuum", "revenue": 499.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Dyson V8 Absolute", "revenue": 399.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Educated by Tara Westover", "revenue": 28, "demand": 4, "initial_inventory": 20 },
    { "name": "Estee Lauder Advanced Night Repair", "revenue": 105, "demand": 2, "initial_inventory": 10 },
    { "name": "Eufy RoboVac 11S", "revenue": 219.99, "demand": 5, "initial_inventory": 20 },
    { "name": "Fenty Beauty Killawatt Highlighter", "revenue": 36, "demand": 2, "initial_inventory": 10 },
    { "name": "First Aid Beauty Ultra Repair Cream", "revenue": 34, "demand": 3, "initial_inventory": 10 },
    { "name": "Fitbit Charge 5", "revenue": 129.99, "demand": 3, "initial_inventory": 10 },
    { "name": "Fitbit Inspire 2", "revenue": 99.95, "demand": 3, "initial_inventory": 10 },
    { "name": "Fitbit Luxe", "revenue": 149.95, "demand": 3, "initial_inventory": 10 },
    { "name": "Fitbit Versa 3", "revenue": 229.95, "demand": 5, "initial_inventory": 20 },
    { "name": "Forever 21 Graphic Tee", "revenue": 12.99, "demand": 7, "initial_inventory": 30 },
    { "name": "Fresh Sugar Lip Treatment", "revenue": 24, "demand": 2, "initial_inventory": 10 },
    { "name": "Gap 1969 Original Fit Jeans", "revenue": 59.99, "demand": 5, "initial_inventory": 20 },
    { "name": "Gap Crewneck Sweatshirt", "revenue": 34.99, "demand": 6, "initial_inventory": 20 },
    { "name": "Gap Essential Crewneck T-Shirt", "revenue": 19.99, "demand": 8, "initial_inventory": 30 },
    { "name": "Gap High Rise Skinny Jeans", "revenue": 49.99, "demand": 5, "initial_inventory": 20 },
    { "name": "Garmin Edge 530", "revenue": 299.99, "demand": 3, "initial_inventory": 10 },
    { "name": "Garmin Fenix 6X Pro", "revenue": 999.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Garmin Forerunner 245", "revenue": 299.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Garmin Forerunner 945", "revenue": 499.99, "demand": 5, "initial_inventory": 20 },
    { "name": "GlamGlow Supermud Clearing Treatment", "revenue": 59, "demand": 2, "initial_inventory": 10 },
    { "name": "Glossier Boy Brow", "revenue": 16, "demand": 3, "initial_inventory": 10 },
    { "name": "Glossier Cloud Paint", "revenue": 18, "demand": 2, "initial_inventory": 10 },
    { "name": "GoPro HERO10 Black", "revenue": 399.99, "demand": 4, "initial_inventory": 20 },
    { "name": "GoPro HERO9 Black", "revenue": 449.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Gone Girl by Gillian Flynn", "revenue": 22.99, "demand": 3, "initial_inventory": 10 },
    { "name": "Google Nest Hub Max", "revenue": 229.99, "demand": 3, "initial_inventory": 10 },
    { "name": "Google Nest Wifi Router", "revenue": 169, "demand": 2, "initial_inventory": 10 },
    { "name": "Google Pixel 6 Pro", "revenue": 899.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Google Pixelbook Go", "revenue": 649.99, "demand": 2, "initial_inventory": 10 },
    { "name": "H&M Slim Fit Jeans", "revenue": 39.99, "demand": 5, "initial_inventory": 20 },
    { "name": "HP Spectre x360 Laptop", "revenue": 1599.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Hamilton Beach FlexBrew Coffee Maker", "revenue": 89.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Hanes ComfortSoft T-Shirt", "revenue": 9.99, "demand": 15, "initial_inventory": 50 },
    { "name": "Harry Potter and the Sorcerer's Stone", "revenue": 24.99, "demand": 4, "initial_inventory": 20 },
    { "name": "Hydro Flask Standard Mouth Water Bottle", "revenue": 32.95, "demand": 5, "initial_inventory": 20 },
    { "name": "Hydro Flask Wide Mouth Water Bottle", "revenue": 39.95, "demand": 6, "initial_inventory": 20 },
    { "name": "Hyperice Hypervolt Massager", "revenue": 349, "demand": 2, "initial_inventory": 10 },
    { "name": "Instant Pot Duo", "revenue": 89.99, "demand": 4, "initial_inventory": 20 },
    { "name": "Instant Pot Duo Crisp", "revenue": 179.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Instant Pot Duo Evo Plus", "revenue": 139.99, "demand": 3, "initial_inventory": 10 },
    { "name": "Instant Pot Duo Nova", "revenue": 99.95, "demand": 2, "initial_inventory": 10 },
    { "name": "Instant Pot Ultra", "revenue": 139.99, "demand": 2, "initial_inventory": 10 },
    { "name": "Keurig K-Elite Coffee Maker", "revenue": 189.99, "demand": 5, "initial_inventory": 20 },
    { "name": "Keurig K-Mini Coffee Maker", "revenue": 79.99, "demand": 3, "initial_inventory": 10 },
    { "name": "Kiehl's Midnight Recovery Concentrate", "revenue": 82, "demand": 2, "initial_inventory": 10 },
    { "name": "Kindle Paperwhite", "revenue": 129.99, "demand": 3, "initial_inventory": 10 },
    { "name": "KitchenAid Artisan Stand Mixer", "revenue": 499.99, "demand": 2, "initial_inventory": 10 },
    { "name": "KitchenAid Stand Mixer", "revenue": 379.99, "demand": 2, "initial_inventory": 10 },
    { "name": "L'Occitane Shea Butter Hand Cream", "revenue": 29, "demand": 3, "initial_inventory": 10 },
    { "name": "L'Oreal Revitalift Serum", "revenue": 39.99, "demand": 3, "initial_inventory": 10 },
    { "name": "LG OLED TV", "revenue": 1299.99, "demand": 3, "initial_inventory": 10 },
    { "name": "La Mer Cr_¨me de la Mer Moisturizer", "revenue": 190, "demand": 2, "initial_inventory": 10 },
    { "name": "Lancome La Vie Est Belle", "revenue": 102, "demand": 2, "initial_inventory": 10 },
    { "name": "Laneige Water Sleeping Mask", "revenue": 25, "demand": 2, "initial_inventory": 10 },
    { "name": "Levi's 501 Jeans", "revenue": 69.99, "demand": 5, "initial_inventory": 20 },
    { "name": "Levi's 511 Slim Fit Jeans", "revenue": 59.99, "demand": 5, "initial_inventory": 20 },
    { "name": "Levi's Sherpa Trucker Jacket", "revenue": 98, "demand": 3, "initial_inventory": 10 },
    { "name": "Levi's Trucker Jacket", "revenue": 89.99, "demand": 3, "initial_inventory": 10 }
  ]
}
```