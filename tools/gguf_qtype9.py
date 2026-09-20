# GGUF v3 probe fix2: correct metadata type sizes.
import struct, sys
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 24: "BF16"}
def rd(fmt):
  return struct.unpack(fmt, f.read(struct.calcsize(fmt)))
f = open(sys.argv[1], "rb")
assert f.read(4) == b"GGUF"
ver, nt, na = rd("<III")
SZ = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
def skip_val(ty):
  global f
  if ty == 8:
    ln2, = rd("<Q")
    f.read(ln2)
  elif ty == 9:
    et, = rd("<I")
    n, = rd("<Q")
    if et == 8:
      for q in range(n):
        ln3, = rd("<Q")
        f.read(ln3)
    else:
      f.read(n * SZ.get(et, 8))
  else:
    f.read(SZ.get(ty, 8))
for i in range(na):
  ln, = rd("<Q")
  k = f.read(ln)
  ty, = rd("<I")
  skip_val(ty)
print("tensors", nt)
for i in range(nt):
  ln, = rd("<Q")
  nm = f.read(ln).decode()
  nd, = rd("<I")
  dims = rd("<%dQ" % nd)
  ty, = rd("<I")
  off, = rd("<Q")
  if "weight" in nm:
    print(nm, GGML_TYPES.get(ty, ty), dims)
