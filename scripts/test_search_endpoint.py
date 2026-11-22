import requests
import json

try:
    res = requests.post('http://127.0.0.1:8000/vae/search_similar', data={'image_url': '/alucard_dataset_image/200319', 'top_k': 4, 'category_filter': 'All'})
    print('Status:', res.status_code)
    if res.status_code == 200:
        data = res.json()
        print('Total indexed:', data.get('total_indexed'))
        print('Matches count:', len(data.get('matches', [])))
        for m in data.get('matches', []):
            print(f"  Match #{m['raw_id']} | Split: {m.get('split')} | Cat: {m.get('category')} | Cosine: {m.get('cosine_similarity')}")
    else:
        print('Response:', res.text)
except Exception as e:
    print('Exception:', e)
