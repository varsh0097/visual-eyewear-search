import pandas as pd
import os
import requests
from concurrent.futures import ThreadPoolExecutor

# --- CONFIG ---
FILE_PATH = r"C:\Users\user\Desktop\A1.0_data_product_images.xlsx"
BASE_OUTPUT_FOLDER = os.path.join("data", "raw_images")

def download_image(url, folder_path, filename):
    if not url or pd.isna(url) or str(url).strip() == "":
        return
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            with open(os.path.join(folder_path, filename), 'wb') as f:
                f.write(response.content)
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def process_product_row(row):
    product_id = str(row['Product Id']).split('.')[0]
    product_folder = os.path.join(BASE_OUTPUT_FOLDER, product_id)
    os.makedirs(product_folder, exist_ok=True)

    image_cols = [c for c in row.index if 'Image' in str(c) and 'Count' not in str(c)]
    for i, col in enumerate(image_cols):
        url = row[col]
        ext = 'jpg'
        filename = f"{product_id}_angle_{i+1}.{ext}"
        download_image(url, product_folder, filename)

def main():
    os.makedirs(BASE_OUTPUT_FOLDER, exist_ok=True)
    df = pd.read_excel(FILE_PATH)
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_product_row, [row for _, row in df.iterrows()])

if __name__ == "__main__":
    main()
