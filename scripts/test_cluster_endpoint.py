import requests
import json

res = requests.post('http://127.0.0.1:8000/vae/cluster_assets', data={'urls': '/alucard_dataset_image/200319'})
print('Status:', res.status_code)
if res.status_code == 200:
    d = res.json()
    print('Silhouette Score:', d.get('silhouette_score'))
    print('Cluster items:', len(d.get('clusters', [])))
    for it in d.get('clusters', [])[:8]:
        print(' ', it['cluster_name'], it['category'], it['url'], it['split'], it['raw_id'])
else:
    print('Response:', res.text)
