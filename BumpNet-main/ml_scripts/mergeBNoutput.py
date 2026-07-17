import numpy as np
import argparse, os
import yaml


def merge_predictions(list_of_tests, test_dir_merge, fileName):
    VALUES=[]
    VALUES2=[]
    for test_dir in list_of_tests:
        if not os.path.exists( f'{test_dir}/{fileName}.npy'):
            print (f'File {test_dir}/{fileName}.npy does not exist, skipping...')
            continue
        
        values = np.load( f'{test_dir}/{fileName}.npy', allow_pickle=True)
        maxlen = max(v.shape[0] for v in values)
        values2 = np.array([np.pad(v, (0, maxlen - v.shape[0]), constant_values=0)  for v in values], dtype=float)
        VALUES.append(values)
        VALUES2.append(values2)
    VALUES = np.stack(VALUES)
    VALUES2 = np.stack(VALUES2)

    # compute the mean and std deviation
    mean = np.mean(VALUES, axis=0)
    std  = np.std( VALUES2, axis=0)
    std = np.array([np.array(row[:len(row2)]) for row,row2 in zip(std, mean)], dtype=object)

    np.save(f'{test_dir_merge}/{fileName}.npy', mean)
    np.save(f'{test_dir_merge}/{fileName}_std.npy', std)
    print (f'Saved merged predictions to {test_dir_merge}/{fileName}.npy')


def mergeBNoutput(config):

    print(f'... Creating {config["output_dir"]}')
    os.system(f'mkdir -p {config["output_dir"]}')

    print(f'... Merging pred_z')
    merge_predictions(config["input_dirs"], f'{config["output_dir"]}', 'pred_z')
    print(f'... Merging pred_b')
    merge_predictions(config["input_dirs"], f'{config["output_dir"]}', 'pred_b')

def main():

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config', help='Config file', default='configs/default.yaml')
    arg_parse.add_argument('--output_dir', help='output directory')
    arg_parse.add_argument('--input_dirs', help='list of training to be merged', nargs='*')
    
    args = arg_parse.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config = config["mergeBNoutput"]

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    
    if args.input_dirs is not None:
        config["input_dirs"] = args.input_dirs
    
    mergeBNoutput(config)

if __name__ == '__main__':
    main()    
