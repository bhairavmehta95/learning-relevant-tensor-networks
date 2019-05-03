import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Learning Relevant Features (Stoudemire 2018)')
    parser.add_argument('--logdir', type=str, default='saved-models/', help='Default log directory')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to append to file')
    parser.add_argument('--parser_type', type=str, default='default', choices=['default', 
        'row', 'column', 'spiral', 'block'], help='Image parser')
    parser.add_argument('--feature_type', type=str, default='default', choices=['default', 
        'cossin'], help='The local feature map to apply')
    parser.add_argument('--dataset', type=str, default='mnist', 
        choices=['mnist', 'fashion', 'hasy'], help='Which dataset to use')
    parser.add_argument('--filename', type=str, help='Default file to load')
    parser.add_argument('--eps', type=float, default=1e-4, help='Truncation epsilon')
    parser.add_argument('--batch-size', type=int, default=2500, help='Batch size for MNIST')
    parser.add_argument('--seed', type=int, default=123, help='Seed')
    parser.add_argument('--nworkers', type=int, default=4, help='Number of multiprocessing workers')
    parser.add_argument('--mtl', action='store_true')

    args = parser.parse_args()

    assert args.filename, 'Must provide a filename to save'

    return args

def get_mtl_args():
    parser = argparse.ArgumentParser(description='MTL Args for Learning Relevant Features (Stoudemire 2018)')
    parser.add_argument('--mixing-mu', type=float, default=0.5, help='Mixing mu for MTL')
    parser.add_argument('--diff-datasets', action='store_true', help='If true, use different datasets')