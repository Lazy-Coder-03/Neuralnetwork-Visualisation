let WIDTH = 800;
let HEIGHT = 600;
let nn, nnv;
const ERROR_THRESHOLD = 0.001;
let currentInputIndex = 0;
let datasets;
let currentDatasetName = 'Encoder 3bit'; // Updated to match the new key name
let isTraining = false;
let epochsPerFrame = 100;

let inputValElem, predictedOutputValElem, targetOutputValElem, selectedNodeInfoElem, trainingMessageElem;
let datasetSelect, trainBtn, epochsSlider, epochsVal, hiddenLayersContainer, addLayerBtn, updateArchBtn, activationFunctionsContainer;
let testDataSelect;

const activationFunctionNames = ['sigmoid', 'relu', 'tanh', 'identity', 'softmax'];

function preload() {
  datasets = loadJSON('trainingData.json');
}

function setup() {
  createCanvas(WIDTH, HEIGHT).parent('p5-canvas-container');
  frameRate(60);

  // Get DOM elements
  inputValElem = document.getElementById('input-val');
  predictedOutputValElem = document.getElementById('predicted-output-val');
  targetOutputValElem = document.getElementById('target-output-val');
  selectedNodeInfoElem = document.getElementById('selected-node-info');
  datasetSelect = document.getElementById('dataset-select');
  trainBtn = document.getElementById('play-pause-btn');
  epochsSlider = document.getElementById('epochs-slider');
  epochsVal = document.getElementById('epochs-val');
  hiddenLayersContainer = document.getElementById('hidden-layers-container');
  addLayerBtn = document.getElementById('add-layer-btn');
  updateArchBtn = document.getElementById('update-arch-btn');
  trainingMessageElem = document.getElementById('training-message-container');
  testDataSelect = document.getElementById('test-data-select');
  activationFunctionsContainer = document.getElementById('activation-functions-container');

  trainBtn.textContent = 'Train Network';

  // Add event listeners
  datasetSelect.addEventListener('change', handleDatasetChange);
  trainBtn.addEventListener('click', toggleTraining);
  epochsSlider.addEventListener('input', () => {
    epochsPerFrame = parseInt(epochsSlider.value);
    epochsVal.textContent = epochsPerFrame;
  });
  addLayerBtn.addEventListener('click', addHiddenLayerControls);
  updateArchBtn.addEventListener('click', updateArchitecture);
  testDataSelect.addEventListener('change', (event) => {
    currentInputIndex = event.target.value;
    updateNodeInfoPanel({ type: 'none' });
  });

  // Populate the dataset dropdown with keys from the JSON
  populateDatasetSelect();

  // Set the initial dropdown value to match the new default name
  datasetSelect.value = currentDatasetName;

  // Initialize with default layers
  const defaultLayers = datasets[currentDatasetName].network.hiddenLayers;
  const defaultActivations = datasets[currentDatasetName].network.options.activationFunctions;
  defaultLayers.forEach((numNodes, index) => addHiddenLayerControls(numNodes, defaultActivations[index]));
  addOutputLayerControls(defaultActivations[defaultActivations.length - 1]);
  setupNetwork(currentDatasetName);
  populateTestDataSelect();
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

// Function to dynamically populate the dataset dropdown
function populateDatasetSelect() {
  for (const key in datasets) {
    if (datasets.hasOwnProperty(key)) {
      const option = document.createElement('option');
      option.value = key; // Use the key as the value
      option.textContent = key; // Use the key as the display name
      datasetSelect.appendChild(option);
    }
  }
}

function stopTraining(message) {
  isTraining = false;
  trainBtn.textContent = 'Train Network';
  trainBtn.classList.remove('bg-red-600', 'hover:bg-red-700');
  trainBtn.classList.add('bg-blue-600', 'hover:bg-blue-700');
  datasetSelect.disabled = false;
  updateArchBtn.disabled = false;
  addLayerBtn.disabled = false;
  document.querySelectorAll('.remove-layer-btn').forEach(btn => btn.disabled = false);
  testDataSelect.disabled = false;
  displayTrainingMessage(message);
}

function displayTrainingMessage(message) {
  trainingMessageElem.innerHTML = `<div class="bg-red-500 text-white p-2 rounded-md my-4">${message}</div>`;
  setTimeout(() => {
    trainingMessageElem.innerHTML = '';
  }, 2000);
}

function setupNetwork(datasetName) {
  const config = datasets[datasetName].network;
  nn = new NeuralNetwork(config.inputNodes, config.hiddenLayers, config.outputNodes, config.options);
  nnv = new NNvisual(WIDTH / 2, HEIGHT / 2, 750, 550, nn, { drawmode: 'center' });
}

function handleDatasetChange(event) {
  if (!isTraining) {
    currentDatasetName = event.target.value;
    while (hiddenLayersContainer.firstChild) {
      hiddenLayersContainer.removeChild(hiddenLayersContainer.firstChild);
    }
    while (activationFunctionsContainer.firstChild) {
      activationFunctionsContainer.removeChild(activationFunctionsContainer.firstChild);
    }
    const defaultLayers = datasets[currentDatasetName].network.hiddenLayers;
    const defaultActivations = datasets[currentDatasetName].network.options.activationFunctions;
    defaultLayers.forEach((numNodes, index) => addHiddenLayerControls(numNodes, defaultActivations[index]));
    addOutputLayerControls(defaultActivations[defaultActivations.length - 1]);
    setupNetwork(currentDatasetName);
    currentInputIndex = 0;
    populateTestDataSelect();
    updateNodeInfoPanel({ type: 'none' });
  } else {
    datasetSelect.value = currentDatasetName;
    displayTrainingMessage('Please pause training before changing the dataset.');
  }
}

function populateTestDataSelect() {
  while (testDataSelect.firstChild) {
    testDataSelect.removeChild(testDataSelect.firstChild);
  }
  const currentData = datasets[currentDatasetName].data;
  currentData.forEach((data, index) => {
    const option = document.createElement('option');
    option.value = index;
    option.textContent = `Test Case ${index + 1}: [${data.inputs}]`;
    testDataSelect.appendChild(option);
  });
}

function toggleTraining() {
  isTraining = !isTraining;
  if (isTraining) {
    trainBtn.textContent = 'Pause Training';
    trainBtn.classList.remove('bg-blue-600', 'hover:bg-blue-700');
    trainBtn.classList.add('bg-red-600', 'hover:bg-red-700');
    datasetSelect.disabled = true;
    updateArchBtn.disabled = true;
    addLayerBtn.disabled = true;
    document.querySelectorAll('.remove-layer-btn').forEach(btn => btn.disabled = true);
    testDataSelect.disabled = true;
  } else {
    stopTraining('Training paused.');
  }
}

function createActivationFunctionSelect(layerType, defaultValue) {
  const select = document.createElement('select');
  select.className = "rounded px-2 py-1 bg-gray-700 text-gray-200";

  activationFunctionNames.forEach(name => {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    if (name === defaultValue) {
      option.selected = true;
    }
    select.appendChild(option);
  });
  return select;
}

function addHiddenLayerControls(defaultNodes = 8, defaultActivation = 'tanh') {
  const layerCount = hiddenLayersContainer.children.length + 1;
  const div = document.createElement('div');
  div.classList.add('layer-input-group');
  div.innerHTML = `
        <label class="text-gray-400 text-sm">Hidden Layer ${layerCount}:</label>
        <input type="number" value="${defaultNodes}" min="1" max="20" class="rounded px-2 py-1 bg-gray-700 text-gray-200 w-20">
        <button class="remove-layer-btn bg-red-500 hover:bg-red-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm leading-none transition-colors duration-200">
            &times;
        </button>
    `;
  hiddenLayersContainer.appendChild(div);

  const selectGroup = document.createElement('div');
  selectGroup.className = 'layer-input-group';
  const selectLabel = document.createElement('label');
  selectLabel.className = "text-gray-400 text-sm";
  selectLabel.textContent = `Hidden Layer ${layerCount}:`;
  const select = createActivationFunctionSelect('hidden', defaultActivation);
  selectGroup.appendChild(selectLabel);
  selectGroup.appendChild(select);

  // Find the last hidden layer activation dropdown and insert before it
  const outputLayerControl = document.getElementById('output-layer-activation');
  activationFunctionsContainer.insertBefore(selectGroup, outputLayerControl);

  div.querySelector('.remove-layer-btn').addEventListener('click', () => {
    if (!isTraining) {
      div.remove();
      selectGroup.remove();
      updateLayerLabels();
    } else {
      displayTrainingMessage('Cannot remove a layer while training.');
    }
  });
}

function addOutputLayerControls(defaultActivation = 'sigmoid') {
  const selectGroup = document.createElement('div');
  selectGroup.id = 'output-layer-activation'; // Add a unique ID
  selectGroup.classList.add('layer-input-group');
  const selectLabel = document.createElement('label');
  selectLabel.className = "text-gray-400 text-sm";
  selectLabel.textContent = `Output Layer:`;
  const select = createActivationFunctionSelect('output', defaultActivation);
  selectGroup.appendChild(selectLabel);
  selectGroup.appendChild(select);
  activationFunctionsContainer.appendChild(selectGroup);
}

function updateLayerLabels() {
  const nodeLabels = hiddenLayersContainer.querySelectorAll('label');
  nodeLabels.forEach((label, index) => {
    label.textContent = `Hidden Layer ${index + 1}:`;
  });
  const activationLabels = activationFunctionsContainer.querySelectorAll('label');
  for (let i = 0; i < activationLabels.length; i++) {
    if (activationLabels[i].textContent.startsWith('Hidden')) {
      activationLabels[i].textContent = `Hidden Layer ${i + 1}:`;
    }
  }
}

function updateArchitecture() {
  if (isTraining) {
    displayTrainingMessage('Cannot update network while training is in progress.');
    return;
  }

  const newHiddenLayers = [];
  const layerInputs = hiddenLayersContainer.querySelectorAll('input[type="number"]');

  if (layerInputs.length === 0) {
    displayTrainingMessage('You must have at least one hidden layer.');
    return;
  }

  for (const input of layerInputs) {
    let parsed = parseInt(input.value);
    if (isNaN(parsed) || parsed < 1 || parsed > 20) {
      parsed = constrain(parsed, 1, 20);
      input.value = parsed;
      displayTrainingMessage('Node count was auto-adjusted to be between 1 and 20.');
    }
    newHiddenLayers.push(parsed);
  }

  const newActivationFunctions = [];
  const activationSelects = activationFunctionsContainer.querySelectorAll('select');
  // First, get the hidden layer activation functions
  for (let i = 0; i < layerInputs.length; i++) {
    newActivationFunctions.push(activationSelects[i].value);
  }
  // Then, get the output layer activation function
  newActivationFunctions.push(activationSelects[activationSelects.length - 1].value);

  const currentDataset = datasets[currentDatasetName];
  const inputCount = currentDataset.network.inputNodes;
  const outputCount = currentDataset.network.outputNodes;
  const options = {
    ...currentDataset.network.options,
    activationFunctions: newActivationFunctions
  };

  nn = new NeuralNetwork(inputCount, newHiddenLayers, outputCount, options);
  nnv = new NNvisual(WIDTH / 2, HEIGHT / 2, 750, 550, nn, { drawmode: 'center' });

  displayTrainingMessage('Network architecture updated successfully!');
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
  if (trainingMessageElem) {
    trainingMessageElem.innerHTML = '';
  }

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
