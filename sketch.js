let WIDTH = 800;
let HEIGHT = 600;
let nn, nnv;
const ERROR_THRESHOLD = 0.001;
let currentInputIndex = 0;
let datasets;
let currentDatasetName = 'encoder3bit';
let isTraining = false;
let epochsPerFrame = 100;

let inputValElem, predictedOutputValElem, targetOutputValElem, selectedNodeInfoElem;
let datasetSelect, trainBtn, epochsSlider, epochsVal, hiddenLayersInput, updateArchBtn;

function preload() {
  datasets = loadJSON('trainingData.json');
}

function setup() {
  createCanvas(WIDTH, HEIGHT).parent('p5-canvas-container');
  frameRate(60);

  setupNetwork(currentDatasetName);

  // Get DOM elements
  inputValElem = document.getElementById('input-val');
  predictedOutputValElem = document.getElementById('predicted-output-val');
  targetOutputValElem = document.getElementById('target-output-val');
  selectedNodeInfoElem = document.getElementById('selected-node-info');
  datasetSelect = document.getElementById('dataset-select');
  trainBtn = document.getElementById('play-pause-btn');
  epochsSlider = document.getElementById('epochs-slider');
  epochsVal = document.getElementById('epochs-val');
  hiddenLayersInput = document.getElementById('hidden-layers-input');
  updateArchBtn = document.getElementById('update-arch-btn');

  trainBtn.textContent = 'Train Network';

  // Add event listeners
  datasetSelect.addEventListener('change', handleDatasetChange);
  trainBtn.addEventListener('click', toggleTraining);
  epochsSlider.addEventListener('input', () => {
    epochsPerFrame = parseInt(epochsSlider.value);
    epochsVal.textContent = epochsPerFrame;
  });
  updateArchBtn.addEventListener('click', updateArchitecture);
}

function draw() {
  background(5);

  if (isTraining) {
    let totalError = 0;
    // Training loop
    for (let i = 0; i < epochsPerFrame; i++) {
      let data = random(datasets[currentDatasetName].data);
      nn.train(data.inputs, data.targets);
    }

    // Check for training completion by calculating total error
    for (let data of datasets[currentDatasetName].data) {
      let outputs = nn.predict(data.inputs);
      totalError += nn.calculateError(outputs, data.targets);
    }

    if (totalError / datasets[currentDatasetName].data.length < ERROR_THRESHOLD) {
      stopTraining('Training complete! Network has learned the pattern.');
    }
  }

  // Draw the network visualization with the user-selected input
  const data = datasets[currentDatasetName].data[currentInputIndex];
  const outputs = nn.predict(data.inputs);
  nnv.show(data.inputs, outputs, currentInputIndex);
  updateDataPanel(data.inputs, outputs, data.targets);
}

function stopTraining(message) {
  isTraining = false;
  trainBtn.textContent = 'Train Network';
  trainBtn.classList.remove('bg-red-600', 'hover:bg-red-700');
  trainBtn.classList.add('bg-blue-600', 'hover:bg-blue-700');
  datasetSelect.disabled = false;
  updateArchBtn.disabled = false; // Enable the button
  displayMessage(message);
}

function displayMessage(message) {
  selectedNodeInfoElem.innerHTML = `<div class="bg-red-500 text-white p-2 rounded-md">${message}</div>`;
}

function setupNetwork(datasetName) {
  const config = datasets[datasetName].network;
  nn = new NeuralNetwork(config.inputNodes, config.hiddenLayers, config.outputNodes, config.options);
  nnv = new NNvisual(WIDTH / 2, HEIGHT / 2, 750, 550, nn, { drawmode: 'center' });
}

function handleDatasetChange(event) {
  if (!isTraining) {
    currentDatasetName = event.target.value;
    setupNetwork(currentDatasetName);
    currentInputIndex = 0;
    updateNodeInfoPanel({ type: 'none' });
  } else {
    datasetSelect.value = currentDatasetName;
    displayMessage('Please pause training before changing the dataset.');
  }
}

function toggleTraining() {
  isTraining = !isTraining;
  if (isTraining) {
    trainBtn.textContent = 'Pause Training';
    trainBtn.classList.remove('bg-blue-600', 'hover:bg-blue-700');
    trainBtn.classList.add('bg-red-600', 'hover:bg-red-700');
    datasetSelect.disabled = true;
    updateArchBtn.disabled = true; // Disable the button
  } else {
    stopTraining('Training paused.');
  }
}

function updateArchitecture() {
  if (isTraining) {
    displayMessage('Cannot update network while training is in progress.');
    return;
  }

  const inputString = hiddenLayersInput.value;
  const hiddenNodes = inputString.split(',').map(num => {
    let parsed = parseInt(num.trim());
    if (isNaN(parsed) || parsed < 1) {
      displayMessage('Invalid input. Please enter a comma-separated list of positive numbers.');
      return -1; // Flag for invalid input
    }
    // Auto-adjust if the number of nodes is more than 20
    if (parsed > 20) {
      parsed = 20;
      displayMessage('Node count was auto-adjusted to 20 for one or more layers.');
    }
    return parsed;
  });

  // Check for any invalid inputs
  if (hiddenNodes.includes(-1)) {
    return; // Stop if there was invalid input
  }

  // Get the current dataset to re-setup the network
  const currentDataset = datasets[currentDatasetName];
  if (!currentDataset) {
    displayMessage('Selected dataset not found.');
    return;
  }

  const inputCount = currentDataset.network.inputNodes;
  const outputCount = currentDataset.network.outputNodes;
  const options = currentDataset.network.options;

  // Update the network configuration
  nn = new NeuralNetwork(inputCount, hiddenNodes, outputCount, options);
  nnv = new NNvisual(WIDTH / 2, HEIGHT / 2, 750, 550, nn, { drawmode: 'center' });

  displayMessage('Network architecture updated successfully!');
}

function keyPressed() {
  if (key === ' ' && !isTraining) {
    currentInputIndex = (currentInputIndex + 1) % datasets[currentDatasetName].data.length;
    // Update selected node info if a node is currently selected
    if (nnv.selectedNode) {
      const data = datasets[currentDatasetName].data[currentInputIndex];
      nn.feedForwardAllLayers(data.inputs);
      const nodeInfo = nnv.getInfoBoxText(nnv.selectedNode.layer, nnv.selectedNode.index, nnv.selectedNode.type);
      updateNodeInfoPanel(nodeInfo);
    } else {
      updateNodeInfoPanel({ type: 'none' });
    }
  }
}

function mousePressed() {
  if (mouseX > 0 && mouseX < width && mouseY > 0 && mouseY < height) {
    const nodeInfo = nnv.mousePressed();
    updateNodeInfoPanel(nodeInfo);
  }
}

function updateDataPanel(inputs, outputs, targets) {
  inputValElem.textContent = `[${inputs}]`;
  predictedOutputValElem.textContent = `[${nf(outputs, 1, 2)}]`;
  targetOutputValElem.textContent = `[${nf(targets, 1, 2)}]`;
}

function updateNodeInfoPanel(nodeInfo) {
  if (nodeInfo.type === 'none') {
    selectedNodeInfoElem.innerHTML = `<p class="text-sm">Click a node to view its details.</p>`;
  } else if (nodeInfo.type === 'node') {
    let html = `<p><strong>Type:</strong> Node</p>`;
    html += `<p><strong>Layer:</strong> ${nodeInfo.layerLabel}</p>`;
    html += `<p><strong>Node Index:</strong> ${nodeInfo.index}</p>`;

    if (nodeInfo.layer > 0) {
      html += `<h4 class="text-gray-400 mt-2">Inputs & Weights:</h4>`;
      let sumTerm = [];
      for (let i = 0; i < nodeInfo.inputDetails.length; i++) {
        const detail = nodeInfo.inputDetails[i];
        sumTerm.push(`[${detail.input} × ${detail.weight}]`);
      }
      html += `<p>${sumTerm.join(' + ')} + Bias: [${nodeInfo.biasValue}]</p>`;
      html += `<hr class="border-t-2 border-dashed border-gray-600 my-2">`;
      html += `<p><strong>Weighted Sum:</strong> ${nodeInfo.weightedSum}</p>`;
      html += `<p><strong>Activation Function:</strong> ${nodeInfo.activationFunctionName}</p>`;
      html += `<p><strong>Final Activation:</strong> ${nf(nodeInfo.activationValue, 1, 4)}</p>`;
    } else {
      html += `<p><strong>Final Activation:</strong> ${nf(nodeInfo.activationValue, 1, 4)}</p>`;
    }
    selectedNodeInfoElem.innerHTML = html;
  } else if (nodeInfo.type === 'bias') {
    let html = `<p><strong>Type:</strong> Bias</p>`;
    html += `<p><strong>Bias Value:</strong> ${nodeInfo.biasValue}</p>`;
    html += `<p><strong>Connected to Layer:</strong> ${nodeInfo.layer}</p>`;
    html += `<h4 class="text-gray-400 mt-2">Bias Weights:</h4>`;
    html += `<ul class="list-disc list-inside">`;
    for (const detail of nodeInfo.biasWeights) {
      html += `<li>Node ${detail.node}: ${detail.weight}</li>`;
    }
    html += `</ul>`;
    selectedNodeInfoElem.innerHTML = html;
  }
}