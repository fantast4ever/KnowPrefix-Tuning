import argparse
import os
import json
from cmudog_generator import data_generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str)
    parser.add_argument("--in_file", type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--out_file', type=str)
    args = parser.parse_args()

    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    args.out_file = os.path.join(args.out_dir, args.out_file)
    with open(args.out_file, 'w', encoding='utf-8') as f:
        for history, response, knowledge in data_generator(args.in_dir, args.in_file, keep_last_n=2):
            f.write(
                json.dumps({
                    'history': history,
                    'response': response,
                    'knowledge': knowledge
                }) + '\n'
            )
