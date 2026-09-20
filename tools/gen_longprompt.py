import sys
n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
sys.stdout.write('The quick brown fox jumps over the lazy dog. ' * n)
