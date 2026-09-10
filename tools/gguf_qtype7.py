# GGUF tensor type probe: prints name + type id for weight tensors.
import struct, sys
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "I8", 17: "I16", 18: "I32", 19: "I64", 24: "BF16"}
p = sys.argv[1]
f = open(p, "rb")
magic = f.read(4)
ver, nt, nkv = struct.unpack("<IQQ", f.read(20))
# skip kv metadata
for _ in range(nkv):
    kl = struct.unpack("<Q", f.read(8))[0]; f.read(kl)
    vt = struct.unpack("<I", f.read(4))[0]
    if vt in (0, 1, 7): f.read(1)
    elif vt in (2, 3): f.read(2)
    elif vt in (4, 5, 6): f.read(4)
    elif vt in (10, 11, 12): f.read(8)
    elif vt == 8:
        sl = struct.unpack("<Q", f.read(8))[0]; f.read(sl)
    elif vt == 9:
        at = struct.unpack("<I", f.read(4))[0]; an = struct.unpack("<Q", f.read(8))[0]
        for __ in range(an):
            if at == 8:
                sl = struct.unpack("<Q", f.read(8))[0]; f.read(sl)
            elif at in (0, 1, 7): f.read(1)
            elif at in (2, 3): f.read(2)
            elif at in (4, 5, 6): f.read(4)
            else: f.read(8)
print("ver", ver, "ntensors", nt)
for i in range(nt):
  ln = struct.unpack("<Q", f.read(8))[0]
  nm = f.read(ln).decode()
  nd = struct.unpack("<I", f.read(4))[0]
  dims = struct.unpack("<%dQ" % nd, f.read(8*nd))
  ty = struct.unpack("<I", f.read(4))[0]
  off = struct.unpack("<Q", f.read(8))[0]
  if "weight" in nm:
    print(nm, GGML_TYPES.get(ty, ty), dims)
