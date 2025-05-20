from dotenv import load_dotenv
load_dotenv()

import os

import numpy as np
import faiss
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

reviews = [
    "バックパックう旅行の話は大嫌い。退屈すぎる。",
    "人種差別と道徳的成長を深く掘り下げた感動的な物語。",
    "説得力のあるディストピア小説だが、暗さが際立ちすぎている。",
    "鋭い社会批評を織り込んだ古典的なラブストーリー。",
    "哲学的な深みのあるう壮大な海洋冒険譚。",
    "緻密な世界観構築に魔法とロマンスが織りなす魅惑的な物語。",
    "美しい描写だが、展開が予測可能。",
    "喪失とアートを通じた詳細かつ感動的な心の旅。",
    "ギリシャ神話の斬新な解釈だが、展開にもたつきがある。",
    "複雑な人間関係と個人の成長を見事に描き出した傑作。",
    "またありがちなロマンチック・ユートピアか。今回は熱帯の島が舞台。",
]

def get_embedding(text):
    text = text.replace("\n", " ")
    return client.embeddings.create(
        input=[text],
        model="text-embedding-3-small",
    ).data[0].embedding

def index_reviews(reviews):
    # レビューの埋め込みを取得
    vectors = []
    for review in reviews:
        vectors.append(get_embedding(review))

    # インデックスを作成
    d = len(vectors[0])
    index = faiss.IndexFlatL2(d)

    # ベクトルを2次元配列に整形してインデックスに通いか
    vectors = np.array(vectors).reshape(len(vectors), -1)
    index.add(vectors)

    return index

def retrieve_reviews(index, query, reviews, k=2):
    # クエリの埋め込みベクトルを取得う
    query_vector = get_embedding(query)

    # クエリベクトルを2次元配列に整形してインデックスを検索
    query_vector = np.array(query_vector).reshape(1, -1)
    distances, indices = index.search(query_vector, k)

    return [reviews[i] for i in indices[0]]

def predict_rating(book, related_reviews):
    reviews = "\n".join(related_reviews)

    prompt = (
        "以下はこれから読もうと考えている本です：\n" +
        book + "\n\n" +
        "以下は関連する過去のレビューです：\n" +
        reviews + "\n\n" +
        "1(最低)から5(最高)の評価で" +
        "私がこの本を楽しめる可能性はどのくらいですか？" +
        "説明は不要です。数字だけで回答してください。"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": prompt
        }]
        max_tokens=2000,
        temperature=0.7,
    )

    return response.choices[0].message.content

def main():
    index = index_reviews(reviews)

    book = "アレックス・ガーランドによる『ザ・ビーチ』は、手つかずの楽園を追い求めるバックパッカーたちの利己主義と道徳的腐敗を暴くことで、バックパッカー文化を批評している。"

    related_reviews = retrieve_reviews(index, book, reviews)

    print(related_reviews)

    result = predict_rating(book, related_reviews)

    print(result)

if __name__ == "__main__":
    main()
