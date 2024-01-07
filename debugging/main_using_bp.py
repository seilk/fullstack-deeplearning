import argparse
from sample_lib import UnionFind

def main(args):
    print(args)
    breakpoint()
    disjoint_set = UnionFind(100)
    disjoint_set.union(0, 3)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg0', action='store_true')
    parser.add_argument('--arg2')
    parser.add_argument('--arg3')
    parser.add_argument('--arg4')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
    