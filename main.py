import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz

# Load model for embedding computation
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# JSON file paths
json_files = [
    "/home/dakshchoudhary/Desktop/truPricer/mergejson/newdata/Blinkit-500085-atta-rice-and-dal-products.json",
    "/home/dakshchoudhary/Desktop/truPricer/mergejson/newdata/Dmart-500085-grocery-products.json",
    "/home/dakshchoudhary/Desktop/truPricer/mergejson/newdata/ZeptoNow-500085-atta-rice-oil-dals-products.json",
]

def load_json_files(files):
    data = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))
    return data

def compute_embeddings(products):
    return {prod['title']: model.encode(prod['title'], convert_to_tensor=True) for prod in products}

def clean_title(title):
    """Generate a cleaned title for grouping similar products"""
    # Remove special characters, convert to lowercase, etc.
    cleaned = title.lower().strip()
    return cleaned

def merge_products_with_embeddings(products, embeddings, similarity_threshold=0.85):
    merged_products = []
    processed_indices = set()
    
    for i, prod1 in enumerate(products):
        if i in processed_indices:
            continue
            
        title1 = prod1['title']
        embedding1 = embeddings[title1]
        
        # Create a new merged product entry
        cleaned_title = clean_title(title1)
        brand = prod1.get('brand', '')
        
        similar_products = [prod1]  # Start with the current product
        processed_indices.add(i)
        
        # Find similar products using embeddings
        for j, prod2 in enumerate(products):
            if j in processed_indices or i == j:
                continue
                
            title2 = prod2['title']
            embedding2 = embeddings[title2]
            
            # Calculate cosine similarity using the embeddings
            similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
            
            if similarity >= similarity_threshold:
                similar_products.append(prod2)
                processed_indices.add(j)
        
        # Create merged product entry
        merged_entry = {
            "cleanedTitle": cleaned_title,
            "brand": brand,
            "products": similar_products
        }
        
        merged_products.append(merged_entry)
    
    return merged_products

# Load and process data
data = load_json_files(json_files)
embeddings = compute_embeddings(data)
merged_data = merge_products_with_embeddings(data, embeddings)

# Save merged JSON
output_file = '/home/dakshchoudhary/Desktop/truPricer/mergejson/outputmerge/merged_grocery_products.json'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)

print(f'Merged data saved to {output_file}')