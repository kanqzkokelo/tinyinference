# GGUF v3 probe: skip metadata, list weight tensors.
import struct, sys
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 24: "BF16"}
def rd(fmt):
  return struct.unpack(fmt, f.read(struct.calcsize(fmt)))
f = open(sys.argv[1], "rb")
assert f.read(4) == b"GGUF"
ver, nt, na = rd("<III")
SZ = {0:4, 1:2, 2:1, 3:1, 4:2, 5:4, 6:4, 7:1, 8:8, 9:8, 10:8, 11:8}
for i in range(na):
  ln, = rd("<Q")
  k = f.read(ln)
  ty, = rd("<I")
  if ty == 8:
    ln2, = rd("<Q")
    f.read(ln2)
  elif ty == 9:
    n, = rd("<Q")
    et, = rd("<I")
    f.read(n * SZ.get(et, 8))
  else:
    f.read(SZ.get(ty, 8))
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
