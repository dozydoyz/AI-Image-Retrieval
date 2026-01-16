import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


CSV_PATH = 'data.csv'
SAVE_DIR = 'gallery_images'
MAX_WORKERS = 128           # 线程数
MAX_IMAGES = 60000          # 下载数量，这里自由调整
TIMEOUT = 3                 # url链接可能会失效，这里设置TIMEOUT,超时就放弃访问

def download_one_image(args):
    url, img_id = args
    save_path = os.path.join(SAVE_DIR, f"{img_id}.jpg")
    if os.path.exists(save_path): return "skipped"
    
    try:
        with requests.Session() as s:
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = s.get(url, headers=headers, timeout=TIMEOUT)
            if resp.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(resp.content)
                return "success"
    except:
        pass
    return "failed"

def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    try:
        df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
        if len(df) > MAX_IMAGES: df = df.iloc[:MAX_IMAGES]
    except Exception as e:
        print(f"读取失败: {e}"); return

    # 自动找 URL 列
    url_col = 'image_url'
    if url_col not in df.columns:
        for col in df.columns:
            if str(df[col].iloc[0]).startswith('http'):
                url_col = col; break
        else: url_col = df.columns[0]

    tasks = [(row[url_col], idx) for idx, row in df.iterrows() if isinstance(row[url_col], str) and row[url_col].startswith('http')]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(download_one_image, tasks), total=len(tasks), unit="img"))

if __name__ == "__main__":
    main()