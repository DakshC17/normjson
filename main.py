import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz

# Load model for embedding computation
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# JSON file paths
json_files = [
    "/home/dakshchoudhary/Desktop/truPricer/mergejson/datasets/Blinkit-500085-atta-rice-and-dal-products.json",
    "/home/dakshchoudhary/Desktop/truPricer/mergejson/datasets/ZeptoNow-500085-atta-rice-oil-dals-products.json",
    "/home/dakshchoudhary/Desktop/truPricer/mergejson/datasets/Dmart-500085-grocery-products.json"
]
def load_json_files(files):
    data = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))
    return data

def compute_embeddings(products):
    return {prod['title']: model.encode(prod['title'], convert_to_tensor=True) for prod in products}

def merge_products(products, embeddings, similarity_threshold=0.85):  
    merged_products = []  
    grouped = set()  
    title_map = {prod["title"]: idx for idx, prod in enumerate(products)}  # Title-to-index mapping  
    titles = list(title_map.keys())  # Unique product titles  

    for i, title1 in enumerate(titles):  
        if i in grouped:  
            continue  

        main_product = products[title_map[title1]]  
        merged_entry = {  
            "title": main_product["title"],  
            "brand": main_product.get("brand", ""),  
            "variants": []  
        }  

        # Use rapidfuzz to find similar product names  
        matches = process.extract(title1, titles, limit=10, scorer=fuzz.ratio)  
        
        for match_title, score, j in matches:  
            if score / 100 >= similarity_threshold and j != i and j not in grouped:  
                matched_product = products[title_map[match_title]]  
                for variant in matched_product["variant"]:
                    variant["platform_url"] = matched_product.get("url", "")  
                    merged_entry["variants"].append(variant)  
                grouped.add(j)  

        for variant in main_product["variant"]:
            variant["platform_url"] = main_product.get("url", "")  
            merged_entry["variants"].append(variant)  
        
        merged_products.append(merged_entry)  
        grouped.add(i)  

    return merged_products  

# Load and process data
data = load_json_files(json_files)
embeddings = compute_embeddings(data)
merged_data = merge_products(data, embeddings)

# Save merged JSON
output_file = '/home/dakshchoudhary/Desktop/truPricer/mergejson/outputmerge/merged_grocery_products.json'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)

print(f'Merged data saved to {output_file}')