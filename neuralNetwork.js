// === Activation Functions and Derivatives ===
function sigmoid(x) {
    if (typeof x !== "number") throw new TypeError("sigmoid expects a number");
    // Prevent overflow
    if (x < -709) return 0; // Math.exp(-709) is smallest nonzero double
    return 1 / (1 + Math.exp(-x));
}

function relu(x) {
    if (typeof x !== "number") throw new TypeError("relu expects a number");
    return Math.max(0, x);
}

function tanh(x) {
    if (typeof x !== "number") throw new TypeError("tanh expects a number");
    return Math.tanh(x);
}

function identity(x) {
    if (typeof x !== "number") throw new TypeError("identity expects a number");
    return x;
}

function softmax(arr) {
    if (!Array.isArray(arr)) throw new TypeError("softmax expects an array");
    if (arr.length === 0) throw new Error("softmax cannot be applied to empty array");

    let max = Math.max(...arr); // numerical stability
    let exps = arr.map(x => Math.exp(x - max));
    let sum = exps.reduce((a, b) => a + b, 0);

    if (sum === 0) {
        console.warn("Warning: softmax denominator was 0, returning uniform distribution");
        return Array(arr.length).fill(1 / arr.length);
    }

    return exps.map(x => x / sum);
}

const activations = { sigmoid, relu, tanh, identity, softmax };

const activation_derivatives = {
    sigmoid: x => {
        let s = sigmoid(x);
        return s * (1 - s);
    },
    relu: x => (x > 0 ? 1 : 0),
    tanh: x => 1 - Math.pow(tanh(x), 2),
    identity: () => 1,
    softmax: () => 1 // crude approximation for backprop, usually handled differently
};

// === NeuralNetwork Class ===
class NeuralNetwork {
    constructor(in_nodes, hidden_layers, out_nodes, options = {}) {
        if (in_nodes instanceof NeuralNetwork) {
            let a = in_nodes;
            this.input_nodes = a.input_nodes;
            this.hidden_layers = Array.isArray(a.hidden_layers) ? [...a.hidden_layers] : [a.hidden_layers];
            this.output_nodes = a.output_nodes;
            this.weights = a.weights.map(w => w.copy());
            this.biases = a.biases.map(b => b.copy());
            this.setLearningRate(a.learning_rate);
            this.setActivationFunctions(a.activation_functions.map(func => func.name));
            this.taskType = a.taskType;
            return;
        }

        // Validate constructor arguments
        if (!Number.isInteger(in_nodes) || in_nodes <= 0) throw new Error("NeuralNetwork: input_nodes must be positive integer");
        if (!Number.isInteger(out_nodes) || out_nodes <= 0) throw new Error("NeuralNetwork: output_nodes must be positive integer");
        if (!Array.isArray(hidden_layers) && !Number.isInteger(hidden_layers)) {
            throw new TypeError("NeuralNetwork: hidden_layers must be array or integer");
        }

        this.input_nodes = in_nodes;
        this.hidden_layers = Array.isArray(hidden_layers) ? hidden_layers : [hidden_layers];
        this.output_nodes = out_nodes;
        this.taskType = options.taskType || 'regression';

        this.weights = [];
        this.biases = [];

        // Input → first hidden
        this.weights.push(new Matrix(this.hidden_layers[0], in_nodes));
        this.biases.push(new Matrix(this.hidden_layers[0], 1));

        // Hidden layers
        for (let i = 0; i < this.hidden_layers.length - 1; i++) {
            this.weights.push(new Matrix(this.hidden_layers[i + 1], this.hidden_layers[i]));
            this.biases.push(new Matrix(this.hidden_layers[i + 1], 1));
        }

        // Last hidden → output
        this.weights.push(new Matrix(out_nodes, this.hidden_layers[this.hidden_layers.length - 1]));
        this.biases.push(new Matrix(out_nodes, 1));

        this.weights.forEach(w => w.randomize());
        this.biases.forEach(b => b.randomize());

        this.setLearningRate(options.learning_rate || 0.01);
        this.setActivationFunctions(options.activationFunctions);

        this.lastInputs = [];
    }

    setLearningRate(learning_rate = 0.01) {
        if (typeof learning_rate !== "number" || learning_rate <= 0) {
            console.warn("Invalid learning_rate provided. Defaulting to 0.01");
            this.learning_rate = 0.01;
        } else {
            this.learning_rate = learning_rate;
        }
    }

    setActivationFunctions(funcNames) {
        const expectedLength = this.hidden_layers.length + 1;

        let functionsToSet = [];
        if (Array.isArray(funcNames) && funcNames.length === expectedLength) {
            functionsToSet = funcNames.map(name => activations[name] || identity);
        } else {
            if (Array.isArray(funcNames) && funcNames.length !== expectedLength) {
                console.warn(
                    `Warning: Expected ${expectedLength} activation functions, got ${funcNames.length}. Defaulting.`
                );
            }

            // Defaults
            let defaultHidden = activations["tanh"];
            let defaultOutput = this.taskType === "classification" ? activations["sigmoid"] : activations["identity"];

            for (let i = 0; i < this.weights.length; i++) {
                if (i < this.weights.length - 1) {
                    functionsToSet.push(defaultHidden);
                } else {
                    functionsToSet.push(defaultOutput);
                }
            }
        }
        this.activation_functions = functionsToSet;
    }

    feedForwardAllLayers(input_array) {
        if (!Array.isArray(input_array)) throw new TypeError("feedForwardAllLayers expects an array of numbers");
        let activations_list = [];
        let current = Matrix.fromArray(input_array);
        activations_list.push(current);

        for (let i = 0; i < this.weights.length; i++) {
            current = Matrix.multiply(this.weights[i], current);
            current.add(this.biases[i]);
            current = current.map(this.activation_functions[i]);
            activations_list.push(current);
        }
        return activations_list;
    }

    predict(input_array) {
        if (!Array.isArray(input_array)) throw new TypeError("predict expects an array of numbers");
        this.lastInputs = input_array;
        let current = Matrix.fromArray(input_array);

        for (let i = 0; i < this.weights.length; i++) {
            current = Matrix.multiply(this.weights[i], current);
            current.add(this.biases[i]);
            current = current.map(this.activation_functions[i]);
        }
        return current.toArray();
    }

    train(input_array, target_array) {
        if (!Array.isArray(input_array) || !Array.isArray(target_array)) {
            throw new TypeError("train expects arrays for input and target");
        }
        if (target_array.length !== this.output_nodes) {
            throw new Error(`train: target length ${target_array.length} does not match output_nodes ${this.output_nodes}`);
        }

        let inputs = Matrix.fromArray(input_array);
        let targets = Matrix.fromArray(target_array);

        let activations_list = [inputs];
        let zs = [];

        let current = inputs;
        for (let i = 0; i < this.weights.length; i++) {
            let z = Matrix.multiply(this.weights[i], current);
            z.add(this.biases[i]);
            zs.push(z);

            current = z.copy();
            current.map(this.activation_functions[i]);
            activations_list.push(current);
        }

        let output = activations_list[activations_list.length - 1];
        let output_errors = (this.taskType === 'classification')
            ? Matrix.subtract(output, targets) // cross-entropy would be better
            : Matrix.subtract(targets, output);

        let error = output_errors;
        for (let l = this.weights.length - 1; l >= 0; l--) {
            let gradient = zs[l].copy();
            gradient.map(activation_derivatives[this.activation_functions[l].name] || (() => 1));

            gradient.multiply(error);
            gradient.multiply(this.learning_rate);

            let prev_activation_T = Matrix.transpose(activations_list[l]);
            let delta_weights = Matrix.multiply(gradient, prev_activation_T);

            this.weights[l].add(delta_weights);
            this.biases[l].add(gradient);

            if (l !== 0) {
                let weights_T = Matrix.transpose(this.weights[l]);
                error = Matrix.multiply(weights_T, error);
            }
        }
    }

    copy() {
        return new NeuralNetwork(this);
    }

    mutate(rate = 0.01) {
        if (typeof rate !== "number" || rate < 0 || rate > 1) {
            throw new Error("mutate: rate must be a number between 0 and 1");
        }
        const mutateFunc = (val) => {
            if (Math.random() < rate) {
                const mutationType = Math.floor(Math.random() * 5);
                switch (mutationType) {
                    case 0: return val + randomGaussian(0, 0.1);
                    case 1: return 0;
                    case 2: return -val;
                    case 3: return Math.random() * 2 - 1;
                    case 4: return val * (0.5 + Math.random());
                }
            }
            return val;
        };

        this.weights.forEach(w => w.map(mutateFunc));
        this.biases.forEach(b => b.map(mutateFunc));
    }

    serialize() {
        return JSON.stringify({
            input_nodes: this.input_nodes,
            hidden_layers: this.hidden_layers,
            output_nodes: this.output_nodes,
            learning_rate: this.learning_rate,
            taskType: this.taskType,
            weights: this.weights.map(w => w.serialize()),
            biases: this.biases.map(b => b.serialize()),
            activation_functions: this.activation_functions.map(f => f.name)
        });
    }

    static deserialize(data) {
        if (typeof data === 'string') data = JSON.parse(data);
        if (!data || !("input_nodes" in data)) {
            throw new Error("NeuralNetwork.deserialize: Invalid data format");
        }
        let nn = new NeuralNetwork(data.input_nodes, data.hidden_layers, data.output_nodes, {
            taskType: data.taskType,
            learning_rate: data.learning_rate
        });
        nn.weights = data.weights.map(w => Matrix.deserialize(w));
        nn.biases = data.biases.map(b => Matrix.deserialize(b));
        nn.setActivationFunctions(data.activation_functions);
        return nn;
    }
}
