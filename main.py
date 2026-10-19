# Drop-in replacement for your notebook from the Word2Vec cell onward.
# Keep: imports, loading the text, tokenization, and the train/val/test split.
# Each "# %%" marks a notebook cell. Hyperparameters are all in cell 1.

# %% Cell 1: settings
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split

WINDOW_SIZE = 5
MIN_COUNT = 2  # train words seen fewer times become <unk>
EMBED_SIZE = 100
HIDDEN1 = 256
HIDDEN2 = 256
KEEP_PROB = 0.7  # dropout: keep 70% of units (set 1.0 to turn off)
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.1
BATCH_SIZE = 128
MAX_EPOCHS = 40
PATIENCE = 3  # stop after this many epochs without val improvement


with open("data/my_sweet_orange_tree.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

# Normalize curly apostrophes and quotation marks
raw_text = raw_text.replace("’", "'").replace("‘", "'")
raw_text = raw_text.replace("“", '"').replace("”", '"')

sentences = sent_tokenize(raw_text.lower())
tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

train_sentences, temp_sentences = train_test_split(
    tokenized_sentences, test_size=0.2, random_state=42
)

val_sentences, test_sentences = train_test_split(
    temp_sentences, test_size=0.5, random_state=42
)


# %% Cell 2: vocabulary with a real <unk> (built from TRAIN only)
UNK_TOKEN = "<unk>"
train_counts = Counter(w for s in train_sentences for w in s)

vocab = [UNK_TOKEN] + [w for w, c in train_counts.most_common() if c >= MIN_COUNT]
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}
unk_idx = word_to_idx[UNK_TOKEN]
VOCAB_SIZE = len(vocab)
PAD_IDX = VOCAB_SIZE  # extra all-zero embedding row used for padding


def to_unk(sentences):
    return [[w if w in word_to_idx else UNK_TOKEN for w in s] for s in sentences]


train_mapped = to_unk(train_sentences)
val_mapped = to_unk(val_sentences)
test_mapped = to_unk(test_sentences)

print("Vocabulary size:", VOCAB_SIZE)

# %% Cell 3: Word2Vec on the mapped train sentences -> embedding table
embedding_model = Word2Vec(
    sentences=train_mapped,
    vector_size=EMBED_SIZE,
    window=5,
    min_count=1,
    sg=1,
    workers=4,
    epochs=10,
    seed=42,
)

E = np.zeros((VOCAB_SIZE + 1, EMBED_SIZE), dtype=np.float32)
for word, idx in word_to_idx.items():
    E[idx] = embedding_model.wv[word]


# %% Cell 4: build (context, target) data
def create_dataset(mapped_sentences):
    ctx_rows, targets = [], []
    for sent in mapped_sentences:
        ids = [word_to_idx[w] for w in sent]
        for i in range(1, len(ids)):
            ctx = ids[max(0, i - WINDOW_SIZE) : i]
            ctx = [PAD_IDX] * (WINDOW_SIZE - len(ctx)) + ctx
            ctx_rows.append(ctx)
            targets.append(ids[i])
    ctx_idx = np.array(ctx_rows, dtype=np.int64)
    X = E[ctx_idx].reshape(len(ctx_idx), -1)  # (n, WINDOW_SIZE * EMBED_SIZE)
    return X, np.array(targets, dtype=np.int64)


X_train, y_train = create_dataset(train_mapped)
X_val, y_val = create_dataset(val_mapped)
X_test, y_test = create_dataset(test_mapped)

print("X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)
print(
    f"<unk> targets  train: {np.mean(y_train == unk_idx):.2%}  "
    f"val: {np.mean(y_val == unk_idx):.2%}  test: {np.mean(y_test == unk_idx):.2%}"
)


# %% Cell 5: the network (with dropout)
def relu(Z):
    return np.maximum(0, Z)


def softmax(Z):
    e = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def initialize_parameters(n_in, n_h1, n_h2, n_out, seed=42):
    rng = np.random.default_rng(seed)

    def he(a, b):
        return (rng.standard_normal((a, b)) * np.sqrt(2 / a)).astype(np.float32)

    return {
        "w1": he(n_in, n_h1),
        "b1": np.zeros((1, n_h1), dtype=np.float32),
        "w2": he(n_h1, n_h2),
        "b2": np.zeros((1, n_h2), dtype=np.float32),
        "w3": he(n_h2, n_out),
        "b3": np.zeros((1, n_out), dtype=np.float32),
    }


def dropout_mask(shape, keep_prob):
    return ((np.random.rand(*shape) < keep_prob) / keep_prob).astype(np.float32)


def forward_propagation(X, p, keep_prob=1.0):
    Z1 = X @ p["w1"] + p["b1"]
    A1 = relu(Z1)
    D1 = None
    if keep_prob < 1.0:
        D1 = dropout_mask(A1.shape, keep_prob)
        A1 = A1 * D1

    Z2 = A1 @ p["w2"] + p["b2"]
    A2 = relu(Z2)
    D2 = None
    if keep_prob < 1.0:
        D2 = dropout_mask(A2.shape, keep_prob)
        A2 = A2 * D2

    A3 = softmax(A2 @ p["w3"] + p["b3"])
    return A3, {"Z1": Z1, "A1": A1, "D1": D1, "Z2": Z2, "A2": A2, "D2": D2, "A3": A3}


def compute_cost(Y, A3):
    m = Y.shape[0]
    correct = np.clip(A3[np.arange(m), Y], 1e-12, 1.0)
    return -np.mean(np.log(correct))


def backpropagation(p, cache, X, Y):
    m = Y.shape[0]
    # softmax + cross-entropy, without building a one-hot matrix
    dZ3 = cache["A3"].copy()
    dZ3[np.arange(m), Y] -= 1
    dW3 = cache["A2"].T @ dZ3 / m
    db3 = dZ3.sum(axis=0, keepdims=True) / m

    dA2 = dZ3 @ p["w3"].T
    if cache["D2"] is not None:
        dA2 = dA2 * cache["D2"]
    dZ2 = dA2 * (cache["Z2"] > 0)
    dW2 = cache["A1"].T @ dZ2 / m
    db2 = dZ2.sum(axis=0, keepdims=True) / m

    dA1 = dZ2 @ p["w2"].T
    if cache["D1"] is not None:
        dA1 = dA1 * cache["D1"]
    dZ1 = dA1 * (cache["Z1"] > 0)
    dW1 = X.T @ dZ1 / m
    db1 = dZ1.sum(axis=0, keepdims=True) / m

    return {"w1": dW1, "b1": db1, "w2": dW2, "b2": db2, "w3": dW3, "b3": db3}


def evaluate(X, y, p, batch_size=1024):
    total_loss, correct = 0.0, 0
    for s in range(0, len(X), batch_size):
        probs, _ = forward_propagation(X[s : s + batch_size], p)  # no dropout
        yb = y[s : s + batch_size]
        total_loss += compute_cost(yb, probs) * len(yb)
        correct += np.sum(np.argmax(probs, axis=1) == yb)
    return total_loss / len(X), correct / len(X)


# %% Cell 6: training with Adam, per-epoch validation, early stopping
params = initialize_parameters(X_train.shape[1], HIDDEN1, HIDDEN2, VOCAB_SIZE)
keys = list(params.keys())
m_adam = {k: np.zeros_like(v) for k, v in params.items()}
v_adam = {k: np.zeros_like(v) for k, v in params.items()}
beta1, beta2, eps = 0.9, 0.999, 1e-8

train_losses, val_losses, val_accs = [], [], []
best_val, best_params, bad_epochs, step = np.inf, None, 0, 0

for epoch in range(1, MAX_EPOCHS + 1):
    order = np.random.permutation(len(X_train))
    for start in range(0, len(X_train), BATCH_SIZE):
        b = order[start : start + BATCH_SIZE]
        Xb, yb = X_train[b], y_train[b]

        A3, cache = forward_propagation(Xb, params, KEEP_PROB)
        grads = backpropagation(params, cache, Xb, yb)

        step += 1
        for k in keys:
            m_adam[k] = beta1 * m_adam[k] + (1 - beta1) * grads[k]
            v_adam[k] = beta2 * v_adam[k] + (1 - beta2) * grads[k] ** 2
            m_hat = m_adam[k] / (1 - beta1**step)
            v_hat = v_adam[k] / (1 - beta2**step)
            if k.startswith("w"):  # decay weights, not biases
                params[k] *= 1 - LEARNING_RATE * WEIGHT_DECAY
            params[k] -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + eps)

    # evaluate WITHOUT dropout so train/val numbers are comparable
    tr_loss, tr_acc = evaluate(X_train, y_train, params)
    va_loss, va_acc = evaluate(X_val, y_val, params)
    train_losses.append(tr_loss)
    val_losses.append(va_loss)
    val_accs.append(va_acc)
    print(
        f"Epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
        f"| val loss {va_loss:.3f} acc {va_acc:.3f}"
    )

    if va_loss < best_val:
        best_val, bad_epochs = va_loss, 0
        best_params = {k: v.copy() for k, v in params.items()}
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"Early stopping: no improvement for {PATIENCE} epochs.")
            break

params = best_params  # roll back to the best epoch
print(f"Restored best weights (val loss {best_val:.3f})")

# %% Cell 7: learning curves + final numbers
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="validation")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

for name, X_, y_ in [
    ("train", X_train, y_train),
    ("val", X_val, y_val),
    ("test", X_test, y_test),
]:
    loss, acc = evaluate(X_, y_, params)
    known = y_ != unk_idx
    k_loss, k_acc = evaluate(X_[known], y_[known], params)
    print(
        f"{name:5s} loss {loss:.3f} acc {acc:.3f} | excluding <unk>: loss {k_loss:.3f} acc {k_acc:.3f}"
    )


# %% Cell 8: prediction and generation (full 5-word context + sampling)
def context_to_input(words):
    ids = [word_to_idx.get(w, unk_idx) for w in words][-WINDOW_SIZE:]
    ids = [PAD_IDX] * (WINDOW_SIZE - len(ids)) + ids
    return E[ids].reshape(1, -1)


def next_word_probs(words):
    probs, _ = forward_propagation(context_to_input(words), params)
    return probs[0]


def show_top_predictions(context, top_n=5):
    words = context.lower().split()
    probs = next_word_probs(words)
    print(f"\nContext: {context}")
    for idx in np.argsort(probs)[::-1][:top_n]:
        print(f"{idx_to_word[idx]:<12} {probs[idx]:.4f}")


def generate_text(start_text, num_words=50, temperature=0.8, top_k=20):
    words = start_text.lower().split()
    for _ in range(num_words):
        probs = next_word_probs(words[-WINDOW_SIZE:]).astype(np.float64)
        probs[unk_idx] = 0.0  # never emit <unk>
        top = np.argsort(probs)[::-1][:top_k]  # keep only top-k words
        p = probs[top] ** (1.0 / temperature)
        p /= p.sum()
        words.append(idx_to_word[int(np.random.choice(top, p=p))])
    return " ".join(words)


show_top_predictions("the")
show_top_predictions("he was")
show_top_predictions("it was a")
print()
print(generate_text("it was a", 30))
