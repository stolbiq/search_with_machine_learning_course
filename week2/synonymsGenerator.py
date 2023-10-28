import fasttext

list_of_words_file = '/workspace/datasets/fasttext/top_words.txt'
model_path = '/workspace/datasets/fasttext/title_model_100.bin'
output_file = '/workspace/datasets/fasttext/synonyms.csv'

with open(list_of_words_file, 'r') as file:
    words = [line[:len(line)-1] for line in file.readlines()]

model = fasttext.load_model(model_path)


def get_synonyms_string(word, model, similarity_threshold=0.75) -> str:
    similar_words = model.get_nearest_neighbors(word)
    synonyms = [word for score, word in similar_words if score >= similarity_threshold]
    return ','.join([word] + synonyms)


with open(output_file, "w") as file:
    for word in words:
        synonyms_string = get_synonyms_string(word, model)
        file.write(synonyms_string + "\n")

