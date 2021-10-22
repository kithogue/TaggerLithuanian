from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import BytePairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from readers.DatasetReader import DatasetReader

resource_path = '/home/ndazhunts/CLARIN/flairLithuanian/datasets/resources/'
train = 'train.txt'
dev = 'dev.txt'
test = 'text.txt'


def run_default():
    return None


def run():
    column_name_map = {0: "token", 1: "upos"}
    corpus: Corpus = ColumnCorpus(resource_path,
                                  column_name_map,
                                  train_file='train.txt',
                                  dev_file='dev.txt',
                                  test_file='text.txt')
    embeddings = BytePairEmbeddings('multi')
    label_type = 'upos'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=label_type)
    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type=label_type,
                            use_crf=True)

    trainer = ModelTrainer(tagger, corpus)
    trainer.train('/datasets/resources/trained-upos',
                  learning_rate=0.1,
                  mini_batch_size=16,
                  max_epochs=15)


def predict():
    sentence = Sentence('Šiandien itin vėjuota: net dideli medžiai griūva, o kai kurie automobiliai sulūžo.')
    model = SequenceTagger.load('/datasets/resources/trained-upos/final-model.pt')
    model.predict(sentence)
    print(sentence.to_tagged_string())


if __name__ == '__main__':
    # DatasetReader.conllu_to_text(resource_path, 'train.conllu', train)
    # DatasetReader.conllu_to_text(resource_path, 'dev.conllu', dev)
    # DatasetReader.conllu_to_text(resource_path, 'test.conllu', test)
    run()

