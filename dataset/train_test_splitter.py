import os 
import shutil
import glob
import argparse

from sklearn.model_selection import train_test_split

def get_filepath_parts(file, input_path):
    filepath, filename = os.path.split(file)
    filepath_parts = filepath.split(os.sep)[len(input_path.split('/')):]
    return filepath_parts, filename

def copy_file(file, new_filepath):
    os.makedirs(os.path.split(new_filepath)[0], exist_ok=True)
    shutil.copy(file, new_filepath)
    print(f"Copied {file} to {new_filepath}")

def handle_unlabeled_files(args):
    files = glob.glob(os.path.join(args.input_path, '*/unlabeled/*.wav'))

    for file in files:
        filepath_parts, filename = get_filepath_parts(file, args.input_path)
        player = filepath_parts[0]

        new_filepath = os.path.join(args.output_path, 'train', 'unlabeled', player, filename)

        copy_file(file, new_filepath)
        
def handle_good_files(args):
    files = glob.glob(os.path.join(args.input_path, '*/labeled/good/**/*.wav'), recursive=True)
    train_files, test_files = train_test_split(files, test_size=args.test_size, random_state=args.seed)

    for file in files:
        filepath_parts, filename = get_filepath_parts(file, args.input_path)
        player = filepath_parts[0]

        if file in train_files:
            new_filepath = os.path.join(args.output_path, 'train', 'labeled', 'good', player, filename)
        else:
            new_filepath = os.path.join(args.output_path, 'test', 'good', player, filename)

        copy_file(file, new_filepath)

def handle_poor_files(args):
    files = glob.glob(os.path.join(args.input_path, '*/labeled/poor/**/*.wav'), recursive=True)
    labels = [os.path.split(file)[0].split(os.sep)[-2] for file in files]
    train_files, test_files, _, _ = train_test_split(files, labels, test_size=args.test_size, random_state=args.seed, stratify=labels)

    for file in files:
        filepath_parts, filename = get_filepath_parts(file, args.input_path)
        player = filepath_parts[0]
        label = filepath_parts[-1]

        if file in train_files:
            new_filepath = os.path.join(args.output_path, 'train', 'labeled', 'poor', label, player, filename)
        else:
            new_filepath = os.path.join(args.output_path, 'test', 'poor', label, player, filename)

        copy_file(file, new_filepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='.')
    parser.add_argument('--output_path', type=str, default='.')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)    
    args = parser.parse_args()

    handle_unlabeled_files(args)
    handle_good_files(args)
    handle_poor_files(args)