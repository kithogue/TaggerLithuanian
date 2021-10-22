import pathlib
from conllu import parse_incr

train_path = "/datasets/resources/train.conllu"
dev_path = "/datasets/resources/dev.conllu"
test_path = "/datasets/resources/test.conllu"


def write_dataset(path: str, start, stop, merged):
    with open(path, "w") as f:
        for i in range(start, stop):
            serialized = merged[i].serialize()
            f.write(serialized)
            f.write('\n\n')


class DatasetReader:

    @staticmethod
    def _concatenate_():
        directory = '/datasets/MATAS-v1.0/CONLLU'
        merged_tokenlists = []
        for filepath in pathlib.Path(directory).glob('**/*'):
            data_file = open(filepath.absolute(), "r", encoding="utf-8")
            for tokenlist in parse_incr(data_file):
                merged_tokenlists.append(tokenlist)
        train_set_len = int(len(merged_tokenlists) * 0.8)
        test_dev_set_len = int((len(merged_tokenlists) - train_set_len) / 2)
        write_dataset(train_path, 0, train_set_len, merged_tokenlists)
        write_dataset(dev_path, train_set_len, train_set_len + test_dev_set_len, merged_tokenlists)
        write_dataset(test_path, train_set_len + test_dev_set_len, len(merged_tokenlists), merged_tokenlists)

    @staticmethod
    def conllu_to_text(path, source_name, target_name):
        sentences = []
        source_file = path + source_name
        with open(source_file, "r") as conllu_file:
            for annotation in parse_incr(conllu_file):
                tokens = [x["form"] for x in annotation]
                upos = [x["upos"] for x in annotation]
                sentence = []
                for pair in zip(tokens, upos):
                    token_tag_tuple = tuple(pair)
                    if token_tag_tuple[1] is not None:
                        sentence.append('\t'.join(token_tag_tuple))
                    else:
                        new_tuple = (token_tag_tuple[0], 'X')
                        sentence.append('\t'.join(new_tuple))
                sentences.append(sentence)
        target_file = path + target_name
        with open(target_file, 'w') as file:
            for sent in sentences:
                file.write('\n'.join(sent))
                file.write('\n\n')
