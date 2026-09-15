import csv, math
from PIL import Image, ImageDraw, ImageFont
CSV = 'bench/scoreboard_decode.csv'
SC = 2
INK = (36,41,47); MUT = (87,96,106); GRID = (208,215,222); PAR = (207,34,46)
BLUE = (9,105,218); PURP = (130,80,223); GRN = (26,127,55); RED = (207,34,46); GREY = (140,149,159)
BG = (255,255,255)
FD = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def F(sz, bold=False):
    return ImageFont.truetype(FB if bold else FD, sz*SC)
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
def save(img, name):
    img.save('docs/assets/' + name)
    print('wrote', name, img.size)
def dash(d, x, y0, y1, fill):
    yy = y0
    while yy < y1:
        d.line([x, yy, x, min(yy+6*SC, y1)], fill=fill, width=2*SC)
        yy += 11*SC
def geomean(rows):
    g = gm([r['ratio'] for r in rows if r['mode'] == 'graph'])
    e = gm([r['ratio'] for r in rows if r['mode'] == 'eager'])
    a = gm([r['ratio'] for r in rows])
    items = [('Graph (CUDA graphs)', g, BLUE), ('Eager (no graphs)', e, PURP), ('Overall geomean', a, GRN)]
    W, H, x0, x1, top, rh, vmax = 760, 250, 250, 690, 78, 48, 1.1
    img = Image.new('RGB', (W*SC, H*SC), BG); d = ImageDraw.Draw(img)
    def X(v): return (x0 + (x1 - x0) * v / vmax) * SC
    d.text((24*SC, 14*SC), 'Decode throughput vs llama.cpp CUDA', fill=INK, font=F(15, True))
    d.text((24*SC, 36*SC), 'geomean over 26 cells - RTX 3050 Laptop 4GB - medians of 5 runs', fill=MUT, font=F(11))
    for v in (0.5, 1.0):
        d.line([X(v), top*SC, X(v), (H-34)*SC], fill=GRID, width=SC)
    dash(d, X(1.0), top*SC, (H-34)*SC, PAR)
    d.text((X(1.0)-34*SC, (H-28)*SC), 'parity 1.0x', fill=MUT, font=F(10))
    for i, it in enumerate(items):
        y = (top + i*rh) * SC
        d.text((24*SC, y+8*SC), it[0], fill=INK, font=F(12))
        d.rounded_rectangle([X(0), y+6*SC, X(it[1]), y+26*SC], radius=4*SC, fill=it[2])
        d.text((X(it[1])+8*SC, y+8*SC), '%.3fx' % it[1], fill=INK, font=F(12, True))
    save(img, 'decode_geomean.png')
def parity(rows):
    cells = sorted([r for r in rows if r['mode'] == 'graph'], key=lambda r: -r['ratio'])
    W, x0, x1, top, rh, vmax = 830, 320, 740, 78, 44, 1.4
    H = top + len(cells)*rh + 38
    img = Image.new('RGB', (W*SC, H*SC), BG); d = ImageDraw.Draw(img)
    def X(v): return (x0 + (x1 - x0) * v / vmax) * SC
    d.text((24*SC, 14*SC), 'Per-cell decode parity - graph mode (ours / oracle)', fill=INK, font=F(15, True))
    d.text((24*SC, 36*SC), 'sorted best to worst - green beats llama.cpp CUDA', fill=MUT, font=F(11))
    for v in (0.5, 1.0):
        d.line([X(v), top*SC, X(v), (H-34)*SC], fill=GRID, width=SC)
    dash(d, X(1.0), top*SC, (H-34)*SC, PAR)
    d.text((X(1.0)-20*SC, (H-28)*SC), 'parity', fill=MUT, font=F(10))
    for i, r in enumerate(cells):
        y = (top + i*rh) * SC
        col = GRN if r['ratio'] >= 1.0 else RED
        d.text((24*SC, y+2*SC), '%s ctx%d %s' % (r['short'], r['ctx'], r['quant']), fill=INK, font=F(11))
        d.text((24*SC, y+19*SC), '%.0f vs %.0f tok/s' % (r['ours'], r['oracle']), fill=MUT, font=F(10))
        d.rounded_rectangle([X(0), y+5*SC, X(r['ratio']), y+27*SC], radius=4*SC, fill=col)
        d.text((X(r['ratio'])+8*SC, y+9*SC), '%.2fx' % r['ratio'], fill=INK, font=F(11, True))
    save(img, 'decode_parity_cells.png')
def tps(rows):
    cells = sorted([r for r in rows if r['mode'] == 'graph'], key=lambda r: (r['short'], r['ctx']))
    W, x0, x1, top, rh = 830, 320, 750, 78, 50
    H = top + len(cells)*rh + 38
    vmax = max(max(r['ours'], r['oracle']) for r in cells) * 1.24
    img = Image.new('RGB', (W*SC, H*SC), BG); d = ImageDraw.Draw(img)
    def X(v): return (x0 + (x1 - x0) * v / vmax) * SC
    d.text((24*SC, 14*SC), 'Decode tok/s - ours (blue) vs llama.cpp CUDA (grey)', fill=INK, font=F(15, True))
    d.text((24*SC, 36*SC), 'graph mode - higher is better', fill=MUT, font=F(11))
    for i, r in enumerate(cells):
        y = (top + i*rh) * SC
        d.text((24*SC, y+14*SC), '%s ctx%d %s' % (r['short'], r['ctx'], r['quant']), fill=INK, font=F(11))
        d.rounded_rectangle([X(0), y+2*SC, X(r['ours']), y+15*SC], radius=3*SC, fill=BLUE)
        d.text((X(r['ours'])+6*SC, y+2*SC), '%.0f' % r['ours'], fill=INK, font=F(10, True))
        d.rounded_rectangle([X(0), y+19*SC, X(r['oracle']), y+32*SC], radius=3*SC, fill=GREY)
        d.text((X(r['oracle'])+6*SC, y+19*SC), '%.0f' % r['oracle'], fill=MUT, font=F(10))
    save(img, 'decode_tps.png')
rows = load()
geomean(rows); parity(rows); tps(rows)
print('cells', len(rows))
