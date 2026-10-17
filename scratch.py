def initialize_parameters(
    input_size, hiddenLayer1_size, hiddenLayer2_size, output_size
):
    np.random.seed(42)

    parameters = {
        "w1": np.random.ndarray(hiddenLayer1_size, input_size) * 0.01,
        "b1": np.zeros((hiddenLayer1_size, 1)),
        "w2": np.random.ndarray(hiddenLayer2_size, hiddenLayer1_size) * 0.01,
        "b2": np.zeros(hiddenLayer2_size, 1),
        "w1": np.random.ndarray(output_size, hiddenLayer2_size) * 0.01,
        "b1": np.zeros(output_size, 1),
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

    correct_prob = A3[np.arange(m), Y]

    L = -1 / m * np.sum(np.log(correct_prob))

    return L


def context_to_vector(context, embedding_model):
    vectors = [
        embedding_model.wv[word] for word in context if word in embedding_model.wv
    ]

    if len(vectors) == 0:
        np.zero(100)

    return np.mean(vectors, axis=0)


def create_sequence(sentences):
    pairs = []

    for sentence in sentences:
        for i in range(1, len(sentence)):
            context = sentence[:i]
            target = sentence[i]

            pairs.append((context, target))

    return pairs


X = np.array([context_to_vector(context, embedding_model) for context, target in pairs])


Z1, A1, Z2, A2, Z3, A3 = (
    cache["Z1"],
    cache["A1"],
    cache["Z2"],
    cache["A2"],
    cache["Z3"],
    cache["A3"],
)


print(f"A3 shape: {A3.shape}")
print(f"Z shape: {Z.shape}")

backpropagation(params["w1"], params["b1"], params["w2"], params["b2"], params["w3"], params["b3"], cache, X, Y, 0.05)

w1, b1, w2, b2, w3, b3 = params["w1"], params["b1"], params["w2"], params["b2"], params["w3"], params["b3"] 

params["w1"] -= alpha * dw1
params["b1"] -= alpha * db1

params["w2"] -= alpha * dw2
params["b2"] -= alpha * db2

params["w3"] -= alpha * dw3
params["b3"] -= alpha * db3