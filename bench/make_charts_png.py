import csv, math
from PIL import Image, ImageDraw, ImageFont
CSV = 'bench/scoreboard_decode.csv'
SC = 2
INK = (31, 35, 40); MUT = (87, 96, 106)
BLUE = (21, 101, 216); PURP = (130, 80, 223); GRN = (26, 127, 55); RED = (207, 34, 46); GREY = (138, 148, 160)
BG = (255, 255, 255)
FD = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def F(sz, bold=False):
    return ImageFont.truetype(FB if bold else FD, sz * SC)
SHORT = {'qwen2.5-0.5b-instruct': 'qwen2.5-0.5b', 'smollm2-135m-instruct': 'smollm2-135m',
         'qwen3-0.6b': 'qwen3-0.6b', 'llama-3.2-1b': 'llama-3.2-1b', 'gemma-4-E2B-it': 'gemma-4-E2B'}
def load():
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        r['ratio'] = float(r['ratio']) if r.get('ratio') and r['ratio'] != 'None' else (float(r['ours_tg_tps']) / float(r['oracle_tg_tps']) if float(r.get('oracle_tg_tps', 0)) > 0 else 1.0); r['ours'] = float(r['ours_tg_tps']); r['oracle'] = float(r['oracle_tg_tps'])
        r['ctx'] = int(r['ctx']); r['short'] = SHORT.get(r['model'], r['model'])
        r['cell'] = '%s ctx%d %s' % (r['short'], r['ctx'], r['quant'])
    return rows
def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))
def save(img, name):
    img.save('docs/assets/' + name)
    print('wrote', name, img.size)
def head(d, title, sub):
    d.text((24 * SC, 14 * SC), title, fill=INK, font=F(17, True))
    d.text((24 * SC, 38 * SC), sub, fill=MUT, font=F(11))
def bar(d, x0, y, x1, h=13, fill=BLUE):
    d.rounded_rectangle([x0, y, x1, y + h * SC], radius=4 * SC, fill=fill)
def geomean(rows):
    g = gm([r['ratio'] for r in rows if r['mode'] == 'graph'])
    e = gm([r['ratio'] for r in rows if r['mode'] == 'eager'])
    a = gm([r['ratio'] for r in rows])
    items = [('Graph (CUDA graphs)', g, BLUE), ('Eager (no graphs)', e, PURP), ('Overall geomean', a, GRN)]
    W, x0, x1, top, rh, vmax = 760, 250, 690, 84, 56, 1.1
    H = top + len(items) * rh + 30
    img = Image.new('RGB', (W * SC, H * SC), BG); d = ImageDraw.Draw(img)
    def X(v): return (x0 + (x1 - x0) * v / vmax) * SC
    head(d, 'Decode throughput vs llama.cpp CUDA', 'geomean over 26 cells - RTX 3050 Laptop 4GB - medians of 5 runs')
    for i, it in enumerate(items):
        y = (top + i * rh) * SC
        d.text((24 * SC, y + 8 * SC), it[0], fill=INK, font=F(11))
        bar(d, X(0), y + 6 * SC, X(it[1]), fill=it[2])
        d.text((X(it[1]) + 8 * SC, y + 6 * SC), '%.3fx' % it[1], fill=INK, font=F(11, True))
    save(img, 'decode_geomean.png')
def parity(rows):
    cells = sorted([r for r in rows if r['mode'] == 'graph'], key=lambda r: -r['ratio'])
    W, x0, x1, top, rh, vmax = 900, 320, 730, 84, 46, 1.55
    H = top + len(cells) * rh + 30
    img = Image.new('RGB', (W * SC, H * SC), BG); d = ImageDraw.Draw(img)
    def X(v): return (x0 + (x1 - x0) * v / vmax) * SC
    head(d, 'Per-cell decode parity - graph mode (ours / oracle)', 'sorted best to worst - green beats llama.cpp CUDA')
    for i, r in enumerate(cells):
        y = (top + i * rh) * SC
        col = GRN if r['ratio'] >= 1.0 else RED
        d.text((24 * SC, y + 8 * SC), r['cell'], fill=INK, font=F(11))
        bar(d, X(0), y + 6 * SC, X(r['ratio']), fill=col)
        d.text((X(r['ratio']) + 8 * SC, y + 6 * SC), '%.2fx' % r['ratio'], fill=INK, font=F(11, True))
        vx = X(r['ratio']) + (8 + 52) * SC
        d.text((vx, y + 6 * SC), '%.0f vs %.0f' % (r['ours'], r['oracle']), fill=MUT, font=F(10))
    save(img, 'decode_parity_cells.png')
def tps(rows):
    cells = sorted([r for r in rows if r['mode'] == 'graph'], key=lambda r: (r['short'], r['ctx']))
    W, x0, x1, top, rh = 830, 320, 750, 84, 50
    H = top + len(cells) * rh + 30
    vmax = max(max(r['ours'], r['oracle']) for r in cells) * 1.24
    img = Image.new('RGB', (W * SC, H * SC), BG); d = ImageDraw.Draw(img)
    def X(v): return (x0 + (x1 - x0) * v / vmax) * SC
    head(d, 'Decode tok/s - ours (blue) vs llama.cpp CUDA (grey)', 'graph mode - higher is better')
    for i, r in enumerate(cells):
        y = (top + i * rh) * SC
        d.text((24 * SC, y + 12 * SC), r['cell'], fill=INK, font=F(11))
        bar(d, X(0), y + 2 * SC, X(r['ours']), fill=BLUE)
        d.text((X(r['ours']) + 6 * SC, y + 2 * SC), '%.0f' % r['ours'], fill=INK, font=F(10, True))
        bar(d, X(0), y + 19 * SC, X(r['oracle']), fill=GREY)
        d.text((X(r['oracle']) + 6 * SC, y + 19 * SC), '%.0f' % r['oracle'], fill=MUT, font=F(10))
    save(img, 'decode_tps.png')
rows = load()
geomean(rows); parity(rows); tps(rows)
print('cells', len(rows))
