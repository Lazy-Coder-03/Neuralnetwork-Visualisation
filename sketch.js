// In sketch.js
WIDTH = 800;
HEIGHT = 600;

let nn, nnv;
const TRAINING_EPOCHS = 1; // Number of training iterations per frame
let currentInputIndex = 0;

// 2. Define the training data for the XOR problem
const trainingData = [
  {
    inputs: [0, 0],
    targets: [0]
  },
  {
    inputs: [0, 1],
    targets: [0]
  },
  {
    inputs: [1, 0],
    targets: [0]
  },
  {
    inputs: [1, 1],
    targets: [1]
  }
];

// Bounding box for the visualization
let x = 10;
let y = 10;
let w = WIDTH - 20; // Corrected width to account for padding on both sides
let h = HEIGHT - 20; // Corrected height to account for padding on both sides

function setup() {
  createCanvas(WIDTH, HEIGHT);
  // Assume NeuralNetwork class is loaded
  nn = new NeuralNetwork(2, [4 ,3], 1, { activationFunctions: ['tanh', 'tanh', 'sigmoid'] });
  // Corrected: Removed the 'r' argument, as it's now calculated dynamically.
  nnv = new NNvisual(x, y, w, h, nn);
  //frameRate(30); // Slow down the frame rate for better visualization of training
}

function draw() {
  background(51);

  // Train the network one time per frame
  for (let i = 0; i < TRAINING_EPOCHS; i++) {
    let data = random(trainingData);
    nn.train(data.inputs, data.targets);
  }

  // Pick one input to visualize based on the current index
  let visualizationInput = trainingData[currentInputIndex].inputs;
  let visualizationTarget = trainingData[currentInputIndex].targets;
  let outputs = nn.predict(visualizationInput);

  // Show the neural network visualization
  nnv.show(visualizationInput, outputs);

  // Display the current outputs and targets for a quick check
  fill(255);
  textSize(16);
  textAlign(LEFT);
  text(`Input: [${visualizationInput}]`, 20, HEIGHT - 60);
  text(`Predicted Output: [${nf(outputs[0], 1, 2)}]`, 20, HEIGHT - 40);
  text(`Target Output: [${nf(visualizationTarget[0], 1, 2)}]`, 20, HEIGHT - 20);
}
function keyPressed() {
  // Check if the pressed key is the spacebar
  if (key === ' ') {
    currentInputIndex = (currentInputIndex + 1) % trainingData.length;
  }
  // Also call the visualization's mousePressed to handle node clicks
}

function mousePressed() {
  nnv.mousePressed();
}