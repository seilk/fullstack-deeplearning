import argparse

def main(args):
    print(args)
    # breakpoint()

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