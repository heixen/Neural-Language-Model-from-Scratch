import numpy as np
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

with open("data/theverdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

sentences = sent_tokenize(raw_text.lower())
tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

print(f"sentences size: {len(sentences)}")
print(f"token size: {len(tokenized_sentences)}")

embedding_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    workers=4,
    epochs=10,
)


WINDOW_SIZE = 3
EMBED_SIZE = embedding_model.wv.vector_size


def create_sequence(sentences, window_size=WINDOW_SIZE):
    pairs = []

    for sentence in sentences:
        for i in range(1, len(sentence)):
            context = sentence[max(0, i - window_size) : i]
            target = sentence[i]
            pairs.append((context, target))

    return pairs


def context_to_vector(context, embedding_model):
    context = context[-WINDOW_SIZE:]

    vectors = [
        embedding_model.wv[word] if word in embedding_model.wv else np.zeros(EMBED_SIZE)
        for word in context
    ]

    # Pad missing positions on the left
    while len(vectors) < WINDOW_SIZE:
        vectors.insert(0, np.zeros(EMBED_SIZE))

    return np.concatenate(vectors)


pairs = create_sequence(tokenized_sentences)

word_to_idx = {word: idx for idx, word in enumerate(embedding_model.wv.index_to_key)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
X = np.array([context_to_vector(context, embedding_model) for context, target in pairs])
y = np.array([target for _, target in pairs])

y = np.array([word_to_idx[word] for word in y])
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


def onehot(Y):
    return np.eye(len(word_to_idx))[Y]


# relu
def relu(Z):
    return np.maximum(0, Z)


# softmax
def softmax(Z):
    exp_z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def initialize_parameters(
    input_size, hiddenLayer1_size, hiddenLayer2_size, output_size
):
    np.random.seed(42)

    parameters = {
        "w1": np.random.randn(input_size, hiddenLayer1_size) * np.sqrt(2 / input_size),
        "b1": np.zeros((1, hiddenLayer1_size)),
        "w2": np.random.randn(hiddenLayer1_size, hiddenLayer2_size)
        * np.sqrt(2 / hiddenLayer1_size),
        "b2": np.zeros((1, hiddenLayer2_size)),
        "w3": np.random.randn(hiddenLayer2_size, output_size)
        * np.sqrt(2 / hiddenLayer2_size),
        "b3": np.zeros((1, output_size)),
    }

    return parameters


def forward_propagation(X, parameters):
    w1, b1, w2, b2, w3, b3 = (
        parameters["w1"],
        parameters["b1"],
        parameters["w2"],
        parameters["b2"],
        parameters["w3"],
        parameters["b3"],
    )

    # first layer
    Z1 = np.dot(X, w1) + b1
    A1 = relu(Z1)

    # second layer
    Z2 = np.dot(A1, w2) + b2
    A2 = relu(Z2)

    # output layer
    Z3 = np.dot(A2, w3) + b3
    A3 = softmax(Z3)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "A3": A3,
    }

    return A3, cache


def compute_cost(Y, A3):
    m = Y.shape[0]

    correct_prob = np.clip(A3[np.arange(m), Y], 1e-12, 1.0)

    L = -1 / m * np.sum(np.log(correct_prob))

    return L


def backpropagation(w1, b1, w2, b2, w3, b3, cache, X, Y):
    m = Y.shape[0]

    Z1, A1 = cache["Z1"], cache["A1"]
    Z2, A2 = cache["Z2"], cache["A2"]
    A3 = cache["A3"]

    # Output layer: softmax + cross-entropy
    dZ3 = A3 - onehot(Y)
    # dZ3 = A3.copy()
    # dZ3[np.arange(len(Y)), Y] -= 1
    dW3 = A2.T @ dZ3 / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    # Hidden layer 2: ReLU
    dA2 = dZ3 @ w3.T
    dZ2 = dA2 * (Z2 > 0)
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Hidden layer 1: ReLU
    dA1 = dZ2 @ w2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3


import numpy as np

# Start with fresh parameters
params = initialize_parameters(X.shape[1], 128, 128, len(word_to_idx))
print("X:", X.shape)
print("w1:", params["w1"].shape)
print("b1:", params["b1"].shape)
print("w2:", params["w2"].shape)
print("b2:", params["b2"].shape)
print("w3:", params["w3"].shape)
print("b3:", params["b3"].shape)


# Adam settings
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
weight_decay = 0.01

keys = ["w1", "b1", "w2", "b2", "w3", "b3"]

m = {k: np.zeros_like(params[k]) for k in keys}
v = {k: np.zeros_like(params[k]) for k in keys}

costs = []
epochs = 1000

for epoch in range(1, epochs + 1):
    A3, cache = forward_propagation(X, params)
    cost = compute_cost(y, A3)
    costs.append(cost)

    grads = backpropagation(
        params["w1"],
        params["b1"],
        params["w2"],
        params["b2"],
        params["w3"],
        params["b3"],
        cache,
        X,
        y,
    )

    grads = dict(zip(keys, grads))

    for k in keys:
        m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
        v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2

        m_hat = m[k] / (1 - beta1**epoch)
        v_hat = v[k] / (1 - beta2**epoch)

        # adaptive learning rate
        # params[k] -= learning_rate * m_hat / (
        #     np.sqrt(v_hat) + epsilon
        # )
        # apply weight decay only on weights matrices
        if k in ["w1", "w2", "w3"]:
            params[k] *= 1 - learning_rate * weight_decay
        params[k] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    if epoch == 1 or epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {cost:.4f}")

# Evaluate training accuracy
A3, _ = forward_propagation(X, params)
predictions = np.argmax(A3, axis=1)

print("Training accuracy:", np.mean(predictions == y))

unique, counts = np.unique(predictions, return_counts=True)
top = np.argsort(counts)[::-1][:10]

print("\nMost predicted words:")
for i in top:
    idx = unique[i]
    print(f"{idx_to_word[idx]}: {counts[i]} ({counts[i]/len(y):.1%})")
