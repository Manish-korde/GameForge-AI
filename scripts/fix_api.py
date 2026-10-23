with open('gui/src/services/api.js', 'r', encoding='utf-8') as f:
    text = f.read()

bt = '`'
text = text.replace('fetch(${BACKEND_URL}/deformation/templates)', f'fetch({bt}${{BACKEND_URL}}/deformation/templates{bt})')
text = text.replace('fetch(${BACKEND_URL}/deformation/detect_skeleton', f'fetch({bt}${{BACKEND_URL}}/deformation/detect_skeleton{bt}')
text = text.replace('fetch(${BACKEND_URL}/deformation/pose', f'fetch({bt}${{BACKEND_URL}}/deformation/pose{bt}')

with open('gui/src/services/api.js', 'w', encoding='utf-8') as f:
    f.write(text)

print('Backticks placed properly.')
