import argparse
import os 

import torch

from data_loader import MyDataLoader
from trainer import ModelTrainer
from pretrainer import ModelPretrainer
from tester import ModelTester
from utils.utils import create_logger, set_seed

def main(args):
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.splitext(args.model_path)[0].replace('/', '.')

    log_filename = os.path.splitext(args.save_path.split('/')[-1])[0]
    if args.pretrain:
        log_filename += '_with-pretrain'
    logger = create_logger(log_filename)

    my_dataloader = MyDataLoader(args.dataset_path, args.batch_size, trace_func=logger.info, sr=args.sr, frame_length=args.frame_length, hop_length=args.hop_length, n_mels=args.n_mels, pitch_shift=args.pitch_shift, random_seed=args.seed)

    logger.info(f"##### Start training. #####")

    if args.pretrain:
        pretrain_dataloader, pretrain_val_dataloader, setting_c_dataloader = my_dataloader.get_dataloader_for_pretrain()

        pretrainer = ModelPretrainer(args, device, model_path, logger)
        pretrainer.train(pretrain_dataloader, pretrain_val_dataloader, setting_c_dataloader)
    
    train_dataloader, val_dataloader = my_dataloader.get_dataloader_for_train()

    trainer = ModelTrainer(args, device, model_path, logger)
    trainer.train(train_dataloader, val_dataloader)

    test_dataloader = my_dataloader.get_dataloader_for_test()

    tester = ModelTester(args.save_path, args.latent_dim, device, model_path=model_path, trace_func=logger.info, save_csvfile=args.csv_path)
    tester.test_dataset(test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../dataset/',
                        help="Dataset path to load.")
    parser.add_argument('--model_path', type=str, default='model.py',
                        help="Model path for training.")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for mini-batch training.")
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Dimension of the latent variable z.')
    parser.add_argument('--eta', type=float, default=1.0,
                        help='Deep SAD hyperparameter eta (must be 0 < eta).')
    parser.add_argument('--save_path', type=str, default='weights/parameters.pth',
                        help='Parameters save path for training.')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help="Number of epochs to train.")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for training.')
    parser.add_argument('--lr_milestones', type=list, default=[30],
                        help='Milestones at which the scheduler multiply the lr by 0.1 for training.')
    parser.add_argument('--patience', type=int, default=20,
                        help="Patience for early stopping of training.")
    parser.add_argument('--es_delta', type=float, default=0,
                        help='Early stopping hyperparameter delta for training.')  
    parser.add_argument('--weight_decay', type=float, default=0.5e-6,
                        help='Weight decay (L2 penalty) hyperparameter for training objective.')
    parser.add_argument('--ae_save_path', type=str, default='weights/pretrained_parameters.pth',
                        help='Parameters save path for pretraining.')   
    parser.add_argument('--ae_num_epochs', type=int, default=100,
                        help="Number of epochs to pretrain autoencoder.")      
    parser.add_argument('--ae_lr', type=float, default=5e-4,
                        help='Learning rate for pretraining autoencoder.')   
    parser.add_argument('--ae_lr_milestones', type=list, default=[50],
                        help='Milestones at which the scheduler multiply the lr by 0.1 for pretraining.')      
    parser.add_argument('--ae_patience', type=int, default=20,
                        help="Patience for early stopping of pretraining.")
    parser.add_argument('--ae_es_delta', type=float, default=0.05,
                        help='Early stopping hyperparameter delta for pretraining.')
    parser.add_argument('--ae_weight_decay', type=float, default=0.5e-3,
                        help='Weight decay (L2 penalty) hyperparameter for pretraining objective.')
    parser.add_argument('--pretrain', action='store_true',
                        help='Pretrain neural network parameters via autoencoder.')
    parser.add_argument('--csv_path', type=str, default='logs/result.csv',
                        help='Csv path to save result.')
    parser.add_argument('--sr', type=int, default=44100,
                        help='Sampling rate.')
    parser.add_argument('--frame_length', type=int, default=1380,
                        help='Window size for the STFT used in extracting the Mel spectrogram.')
    parser.add_argument('--hop_length', type=int, default=345,
                        help='Number of samples between successive frames in the STFT.')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='Number of Mel frequency bands to use when extracting the Mel spectrogram.')
    parser.add_argument('--pitch_shift', type=int, default=5,
                        help='Random pitch shift range.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')

    args = parser.parse_args()

    main(args)

    
