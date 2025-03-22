import numpy as np
import os
k = 8
html = '<html><body>'
html += f'<h1>class={k}</h1>\n'
files = [f for f in os.listdir('.') if f.startswith(f'digits{k}-') and f.endswith('.png')]
for f in files:
    if f.split('-')[3] != '0': continue
    f = '-'.join(f.split('-')[:-3])
    for i in np.linspace(0, 24, 6, dtype=int):
        f2 = [t for t in files if t.startswith(f'{f}-{i}-')][0]
        html += f'<a href="{f2}"><img width="128" height="128" src="{f2}"></a>\n'
    html += '<br>\n'
html += '</body></html>'
print(html)
