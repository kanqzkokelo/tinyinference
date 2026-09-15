import csv, math
CSV = 'bench/scoreboard_decode.csv'
NL = chr(10)
SHORT = {'qwen2.5-0.5b-instruct': 'qwen2.5-0.5b', 'smollm2-135m-instruct': 'smollm2-135m',
         'qwen3-0.6b': 'qwen3-0.6b', 'llama-3.2-1b': 'llama-3.2-1b', 'gemma-4-E2B-it': 'gemma-4-E2B'}
def load():
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        r['ratio'] = float(r['ratio']); r['ours'] = float(r['ours_tg_tps']); r['oracle'] = float(r['oracle_tg_tps'])
        r['ctx'] = int(r['ctx']); r['short'] = SHORT.get(r['model'], r['model'])
    return rows
def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))
def hdr(W, H):
    return '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="Segoe UI,Helvetica,Arial,sans-serif">' % (W, H)
CSS = '<style>.t{fill:#24292f}.m{fill:#57606a}.v{fill:#24292f;font-weight:600}.grid{stroke:#d0d7de;stroke-width:1}.par{stroke:#cf222e;stroke-width:1.5;stroke-dasharray:6 4}</style>'
def bg(W, H):
    return '<rect x="0" y="0" width="%d" height="%d" rx="8" fill="#ffffff" stroke="#d0d7de"/>' % (W, H)
def geomean_chart(rows):
    g = gm([r['ratio'] for r in rows if r['mode'] == 'graph'])
    e = gm([r['ratio'] for r in rows if r['mode'] == 'eager'])
    a = gm([r['ratio'] for r in rows])
    items = [('Graph (CUDA graphs)', g, '#0969da'), ('Eager (no graphs)', e, '#8250df'), ('Overall geomean', a, '#1a7f37')]
    W, x0, x1, top, rh = 760, 220, 700, 64, 44
    H = top + 3 * rh + 40
    vmax = 1.1
    def X(v): return x0 + (x1 - x0) * v / vmax
    s = [hdr(W, H), CSS, bg(W, H)]
    s.append('<text class="t" x="24" y="30" font-size="16" font-weight="700">Decode throughput vs llama.cpp CUDA</text>')
    s.append('<text class="m" x="24" y="50" font-size="12">geomean over 26 cells - RTX 3050 Laptop 4GB - medians of 5 runs post-warmup</text>')
    for v in (0.5, 1.0):
        s.append('<line class="grid" x1="%d" y1="%d" x2="%d" y2="%d"/>' % (X(v), top - 8, X(v), H - 32))
    s.append('<line class="par" x1="%.1f" y1="%d" x2="%.1f" y2="%d"/>' % (X(1.0), top - 8, X(1.0), H - 32))
    s.append('<text class="m" x="%.1f" y="%d" font-size="11" text-anchor="middle">parity 1.0x</text>' % (X(1.0), H - 14))
    for i, item in enumerate(items):
        y = top + i * rh
        s.append('<text class="t" x="24" y="%d" font-size="13">%s</text>' % (y + 20, item[0]))
        s.append('<rect x="%.1f" y="%d" width="%.1f" height="24" rx="4" fill="%s"/>' % (x0, y + 4, X(item[1]) - x0, item[2]))
        s.append('<text class="v" x="%.1f" y="%d" font-size="13">%.3fx</text>' % (X(item[1]) + 8, y + 22, item[1]))
    return NL.join(s) + NL + '</svg>' + NL
def parity_chart(rows):
    cells = sorted([r for r in rows if r['mode'] == 'graph'], key=lambda r: -r['ratio'])
    W, x0, x1, top, rh, vmax = 780, 300, 690, 64, 38, 1.4
    H = top + len(cells) * rh + 40
    def X(v): return x0 + (x1 - x0) * v / vmax
    s = [hdr(W, H), CSS, bg(W, H)]
    s.append('<text class="t" x="24" y="30" font-size="16" font-weight="700">Per-cell decode parity - graph mode (ours / oracle)</text>')
    s.append('<text class="m" x="24" y="50" font-size="12">sorted best to worst - green bars beat llama.cpp CUDA</text>')
    for v in (0.5, 1.0):
        s.append('<line class="grid" x1="%d" y1="%d" x2="%d" y2="%d"/>' % (X(v), top - 8, X(v), H - 32))
    s.append('<line class="par" x1="%.1f" y1="%d" x2="%.1f" y2="%d"/>' % (X(1.0), top - 8, X(1.0), H - 32))
    s.append('<text class="m" x="%.1f" y="%d" font-size="11" text-anchor="middle">parity</text>' % (X(1.0), H - 14))
    for i, r in enumerate(cells):
        y = top + i * rh
        col = '#1a7f37' if r['ratio'] >= 1.0 else '#cf222e'
        lab = '%s ctx%d %s' % (r['short'], r['ctx'], r['quant'])
        s.append('<text class="t" x="24" y="%d" font-size="12">%s</text>' % (y + 19, lab))
        s.append('<text class="m" x="24" y="%d" font-size="10">%.0f vs %.0f tok/s</text>' % (y + 31, r['ours'], r['oracle']))
        s.append('<rect x="%d" y="%d" width="%.1f" height="22" rx="4" fill="%s"/>' % (x0, y + 4, X(r['ratio']) - x0, col))
        s.append('<text class="v" x="%.1f" y="%d" font-size="12">%.2fx</text>' % (X(r['ratio']) + 8, y + 21, r['ratio']))
    return NL.join(s) + NL + '</svg>' + NL
def tps_chart(rows):
    cells = sorted([r for r in rows if r['mode'] == 'graph'], key=lambda r: (r['short'], r['ctx']))
    W, x0, x1, top, rh = 780, 300, 700, 64, 44
    H = top + len(cells) * rh + 40
    vmax = max(max(r['ours'], r['oracle']) for r in cells) * 1.18
    def X(v): return x0 + (x1 - x0) * v / vmax
    s = [hdr(W, H), CSS, bg(W, H)]
    s.append('<text class="t" x="24" y="30" font-size="16" font-weight="700">Decode tok/s - ours (blue) vs llama.cpp CUDA (grey)</text>')
    s.append('<text class="m" x="24" y="50" font-size="12">graph mode - higher is better</text>')
    for i, r in enumerate(cells):
        y = top + i * rh
        lab = '%s ctx%d %s' % (r['short'], r['ctx'], r['quant'])
        s.append('<text class="t" x="24" y="%d" font-size="12">%s</text>' % (y + 26, lab))
        s.append('<rect x="%d" y="%d" width="%.1f" height="14" rx="3" fill="#0969da"/>' % (x0, y + 2, X(r['ours']) - x0))
        s.append('<text class="t" x="%.1f" y="%d" font-size="11">%.0f</text>' % (X(r['ours']) + 6, y + 14, r['ours']))
        s.append('<rect x="%d" y="%d" width="%.1f" height="14" rx="3" fill="#8c959f"/>' % (x0, y + 20, X(r['oracle']) - x0))
        s.append('<text class="m" x="%.1f" y="%d" font-size="11">%.0f</text>' % (X(r['oracle']) + 6, y + 32, r['oracle']))
    return NL.join(s) + NL + '</svg>' + NL
rows = load()
open('docs/assets/decode_geomean.svg', 'w').write(geomean_chart(rows))
open('docs/assets/decode_parity_cells.svg', 'w').write(parity_chart(rows))
open('docs/assets/decode_tps.svg', 'w').write(tps_chart(rows))
print('wrote 3 charts for %d rows' % len(rows))
