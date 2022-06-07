import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout ratio')
    parser.add_argument('--use_pretrain', type=str, default="unuse", help="use or unuse")
    

    # ========================= Data Configs ==========================
    parser.add_argument('--pretrain_annotation', type=str, default='../data/annotations/unlabeled.json')
    parser.add_argument('--train_annotation', type=str, default='../data/annotations/labeled.json')
    parser.add_argument('--test_annotation', type=str, default='../data/annotations/test_a.json')
    parser.add_argument('--pretrain_zip_feats', type=str, default='../data/zip_feats/unlabeled.zip')
    parser.add_argument('--train_zip_feats', type=str, default='../data/zip_feats/labeled.zip')
    parser.add_argument('--test_zip_feats', type=str, default='../data/zip_feats/test_a.zip')
    parser.add_argument('--test_output_csv', type=str, default='../data/result.csv')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=64, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=256, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=256, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")
    
    # ========================= Pretrain Configs ========================== 
    parser.add_argument('--vocab_size', default=21128, type=int, help="bert chinese vocabulary size")
    parser.add_argument('--pretrain_ckpt_file', type=str, default='../save/pretrain/model_epoch_0_loss_3.736_acc0.502.bin')

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='../save/v9')
    parser.add_argument('--saved_pretrain_model_path', type=str, default='../save/pretrain')
    parser.add_argument('--ckpt_file', type=str, default='../save/v8/model_epoch_3_mean_f1_0.6391.bin')
    parser.add_argument('--best_score', default=0.5, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=20, help='How many epochs')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=3e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default="hfl/chinese-macbert-base",
                        help="hfl/chinese-macbert-base, or hfl/chinese-bert-wwm-ext")
    parser.add_argument('--bert_cache', type=str, default='../data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=50)
    parser.add_argument('--bert_learning_rate', type=float, default=2e-5)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.2)

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=8)
    parser.add_argument('--vlad_hidden_size', type=int, default=1024, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=512, help="linear size before final linear")

    return parser.parse_args()
