import argparse

from VietOCR.vietocr.model.trainer import Trainer
from VietOCR.vietocr.tool.config import Cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='see example at ')
    parser.add_argument('--checkpoint', required=False, help='your checkpoint')
    parser.add_argument('--weights', required=False, help='your weights')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    trainer = Trainer(config, pretrained=False)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        
    if args.weights:
        trainer.load_weights(args.weights)

    trainer.train()

if __name__ == '__main__':
    main()
