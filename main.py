import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load model for embedding computation
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def load_json_files(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)

            # Convert JioMart structure into a standard format
            if "jiomart" in file_path.lower():
                file_data = convert_jiomart_format(file_data)

            data.extend(file_data)
    return data

def convert_jiomart_format(jiomart_products):
    """Convert JioMart structure to match other datasets."""
    formatted_products = []
    for product in jiomart_products:
        formatted_products.append({
            "title": product["title"],
            "brand": product.get("brand", ""),
            "pincode": product.get("pincode"),
            "url": product["url"],
            "variant": [{
                "quantity": product.get("quantity", ""),
                "mrp": product.get("mrp", ""),
                "price": product.get("price", ""),
                "articleId": product["article_id"]
            }]
        })
    return formatted_products

def compute_embeddings(products):
    return {prod['title']: model.encode(prod['title'], convert_to_tensor=True) for prod in products}

def clean_title(title):
    return title.lower().strip()

def merge_products(products, embeddings, similarity_threshold=0.90):
    merged_products = []
    processed_indices = set()
    
    for i, prod1 in enumerate(products):
        if i in processed_indices:
            continue

        title1 = prod1['title']
        cleaned_title = clean_title(title1)
        brand = prod1.get('brand', '')
        embedding1 = embeddings[title1]
        
        similar_products = [prod1]
        processed_indices.add(i)

        for j, prod2 in enumerate(products):
            if j in processed_indices or i == j:
                continue
            
            title2 = prod2['title']
            embedding2 = embeddings[title2]
            similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
            
            if similarity >= similarity_threshold:
                similar_products.append(prod2)
                processed_indices.add(j)
        
        merged_entry = {
            "cleanedTitle": cleaned_title,
            "brand": brand,
            "products": []
        }

        variants_map = {}
        for product in similar_products:
            for variant in product.get("variant", []):
                quantity = variant.get("quantity", "")
                article_id = variant.get("articleId", None)
                platform_url = product.get("url", "")
                price = variant.get("price", "")
                mrp = variant.get("mrp", "")
                
                if quantity and article_id is not None:
                    if quantity not in variants_map:
                        variants_map[quantity] = {
                            "quantity": quantity,
                            "mrp": mrp,
                            "prices": []
                        }
                    
                    price_entry = {
                        "articleId": article_id,
                        "platformUrl": platform_url,
                        "price": price
                    }
                    variants_map[quantity]["prices"].append(price_entry)

        if variants_map:
            merged_entry["products"].append({
                "title": title1,
                "variant": list(variants_map.values()),
                "pincode": prod1.get("pincode", None)
            })
            
            merged_products.append(merged_entry)
        else:
            merged_products.append({
                "cleanedTitle": cleaned_title,
                "brand": brand,
                "products": [{
                    "title": title1,
                    "variant": prod1.get("variant", []),
                    "pincode": prod1.get("pincode", None)
                }]
            })
    
    for i, prod in enumerate(products):
        if i not in processed_indices:
            merged_products.append({
                "cleanedTitle": clean_title(prod['title']),
                "brand": prod.get('brand', ''),
                "products": [{
                    "title": prod['title'],
                    "variant": prod.get("variant", []),
                    "pincode": prod.get("pincode", None)
                }]
            })
    
    return merged_products

# File paths (update with actual paths)
json_files = [
    "/home/dakshchoudhary/Desktop/truPricer/mergejson/newdata/Blinkit-500085-atta-rice-and-dal-products.json",
    "/home/dakshchoudhary/Desktop/truPricer/mergejson/newdata/Dmart-500085-grocery-products.json",
    "/home/dakshchoudhary/Desktop/truPricer/mergejson/newdata/ZeptoNow-500085-atta-rice-oil-dals-products.json",
    "/home/dakshchoudhary/Desktop/truPricer/mergejson/newdata/JioMartGroceries_500074_2024-04-14.json",  # JioMart file included
]

data = load_json_files(json_files)
embeddings = compute_embeddings(data)
merged_data = merge_products(data, embeddings)

# Save merged JSON
output_file = '/home/dakshchoudhary/Desktop/truPricer/mergejson/outputmerge/merged_grocery_products.json'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)

print(f'Merged data saved to {output_file}')
