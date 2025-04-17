import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# 🤗 Huggingface Hubから以下のようにしてモデルをダウンロードできます
tokenizer = AutoTokenizer.from_pretrained("pfnet/plamo-embedding-1b", trust_remote_code=True)
model = AutoModel.from_pretrained("pfnet/plamo-embedding-1b", trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

query = "ターミネーター2って面白いですよね！"
documents = [
    "terminator 2 is such a great movie!",
    "私はまどマギの方が面白いと思います。"
]

with torch.inference_mode():
    # 情報検索におけるクエリ文章の埋め込みに関しては、`encode_query` メソッドを用いてください
    # tokenizerも渡す必要があります
    query_embedding = model.encode_query(query, tokenizer)
    # それ以外の文章に関しては、 `encode_document` メソッドを用いてください
    # 情報検索以外の用途についても、 `encode_document` メソッドを用いてください
    document_embeddings = model.encode_document(documents, tokenizer)

# モデルに文章を入力して得られたベクトル間の類似度は、近い文章は高く、遠い文章は低くなります
# この性質を用いて情報検索などに活用することができます
similarities = F.cosine_similarity(query_embedding, document_embeddings)
print(similarities)
# tensor([0.8812, 0.5533])
