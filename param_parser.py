import argparse
import yaml


def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run CapsGNN.")

    def str2bool(x): return x.lower() == "true"

    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help="The name of the dataset.")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--devices', type=int, default=[0],
                        nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--k', type=int, default=3)
    # parser.add_argument('--k', type=int, default=5)
    # parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--k', type=int, default=5)
    # parser.add_argument('--k', type=int, default=30)
    # parser.add_argument('--k', type=int, default=40)
    # parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--k2', type=int, default=3)

    parser.add_argument("--features-dimensions", type=int, default=32,
                        help="node features dimensions. Default is 128.")
    parser.add_argument("--capsule-dimensions", type=int, default=6,
                        help="Capsule dimensions. Default is 4,6,8,10,12.")
    # parser.add_argument("--capsule-num", type=int, default=20,
    #                     help="Graph capsule num. Default is 10.")
    # parser.add_argument("--capsule-num", type=int, default=10)
    # parser.add_argument("--capsule-num", type=int, default=5)

    # parser.add_argument("--capsule-num", type=int, default=15)
    # parser.add_argument("--capsule-num", type=int, default=30)
    # parser.add_argument("--capsule-num", type=int, default=25)
    # parser.add_argument("--capsule-num", type=int, default=20)
    parser.add_argument("--capsule-num", type=int, default=10) #this
    # parser.add_argument("--capsule-num", type=int, default=80)
    # parser.add_argument("--capsule-num", type=int, default=120)
    parser.add_argument("--num-gcn-layers", type=int, default=3),
    parser.add_argument("--num-gcn-channels", type=int, default=2),
    parser.add_argument("--num-iterations", type=int, default=3,
                        help="Number of routing iterations. Default is 3.")
    parser.add_argument("--theta", type=float, default=0.1,
                        help="Reconstruction loss weight. Default is 0.1.")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epoch_select', type=str, default='val_min',
                        help="{test_max, val_min} test_max: select a single epoch; \
                            val_min: select epoch with the lowest val loss.")
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--with_eval_mode', type=str2bool, default=True)

    parser.add_argument('--exp', type=str, default="test")
    parser.add_argument('--data_root', type=str, default="./data")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed. Default is 1.")
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument('--log-path', type=str, default="log/train_loss",
                        help="The path of training log.")

    parser.add_argument(
        '--config',
        # default='./config/ENZYMES/HGCN_ori.yaml',
        help='path to the configuration file')

    return parser


def get_parser():
    # load arg form config file
    parser = parameter_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    return parser.parse_args()
