import math
import time
from openai import OpenAI
from typing import List


client = OpenAI(base_url="http://localhost:8080/v1", api_key= "-")
EMBEDDING_MODEL_NAME =  "text-embeddings-inference"


def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=text
    )
    return response.data[0].embedding


text1 = "พนักงานยิ้มแย้มแจ่มใสบริการดี"
# text2 = "พนักงานหน้าบึ้ง เสริฟอาหารไม่ตรงเวลา"
text2 = "รออาหารนานมาก ไม่ประทับใจอย่างแรง"


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot / (norm1 * norm2)


if __name__ == "__main__":
    start = time.perf_counter()
    embedding1 = get_embedding(text1)
    time1 = time.perf_counter() - start

    start = time.perf_counter()
    embedding2 = get_embedding(text2)
    time2 = time.perf_counter() - start

    start = time.perf_counter()
    similarity = cosine_similarity(embedding1, embedding2)
    time_sim = time.perf_counter() - start

    print(f"text1: {text1}")
    print(f"text2: {text2}")
    print(f"Cosine similarity: {similarity:.4f}")
    print(f"Inference time text1: {time1:.4f} s")
    print(f"Inference time text2: {time2:.4f} s")
    print(f"Total inference time: {time1 + time2:.4f} s")
    print(f"Cosine similarity compute time: {time_sim:.6f} s")
