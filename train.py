from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data


def main():
    # load train dataset
    data = load_coco_data(data_path='./new_data', split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path='./new_data', split='val')

    model = CaptionGenerator(word_to_idx, dim_att=[4,512], dim_feature=[196, 512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    idx_to_word = {v: k for k, v in word_to_idx.iteritems()}

    solver = CaptioningSolver(model, data, val_data, idx_to_word, n_epochs=15, batch_size=64, update_rule='adam',
                                          learning_rate=0.001, print_every=50, save_every=5, image_path='./image/',
                                    pretrained_model=None, model_path='model/lstm2/', test_model='model/lstm2/model-15',
                                     print_bleu=True, log_path='log/')

    solver.train()

if __name__ == "__main__":
    main()
