import argparse

# Arguments Parser for X-2ch

def parse_args():
    parser = argparse.ArgumentParser(description="Run X2CH.")

    # Dataset
    parser.add_argument('--dataset', default='last-fm', help='last-fm, amazon-book')

    # Paths
    parser.add_argument('--data_path', nargs='?', default='../Data/', help='Input data path.')
    parser.add_argument('--weights_path', nargs='?', default='', help='Store model path.')

    # Hyper-Parameters
    parser.add_argument('--epoch', type=int, default=100, help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=64, help='CF Embedding size.')
    parser.add_argument('--kge_size', type=int, default=64, help='KG Embedding size.')
    parser.add_argument('--batch_size', type=int, default=1024, help='CF batch size.')

    # for Ablation Study
    parser.add_argument('--use_kg-edges', type=bool, default=True, help='whether using knowledg-aware edges')
    parser.add_argument('--use_attention', type=bool, default=True, help='whether using channel-wise attention')

    return parser.parse_args()
