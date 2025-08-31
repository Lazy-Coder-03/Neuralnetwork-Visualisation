let WIDTH = 600;
let HEIGHT = 400;
let nn, nnv;
const ERROR_THRESHOLD = 0.001;
let currentInputIndex = 0;
let datasets;
let currentDatasetName = 'Encoder 3bit';
let isTraining = false;
let epochsPerFrame = 100;
const WEIGHT_THRESHOLD = 0.5; // You can adjust this value as needed

let inputValElem, predictedOutputValElem, targetOutputValElem, selectedNodeInfoElem, trainingMessageElem;
let datasetSelect, trainBtn, epochsSlider, epochsVal, hiddenLayersContainer, addLayerBtn, updateArchBtn, activationFunctionsContainer;
let testDataSelect, trainingStatusElem, playIcon, pauseIcon, buttonText;

const activationFunctionNames = ['sigmoid', 'relu', 'tanh', 'identity', 'softmax'];

function preload() {
  datasets = loadJSON('trainingData.json');
}

function setup() {
  // Canvas dimensions are now fixed at 800x600
  const canvas = createCanvas(WIDTH, HEIGHT);
  canvas.parent('p5-canvas-container');
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
  trainingStatusElem = document.getElementById('training-status');
  playIcon = document.getElementById('play-icon');
  pauseIcon = document.getElementById('pause-icon');
  buttonText = document.getElementById('button-text');

  // Add event listeners
  datasetSelect.addEventListener('change', handleDatasetChange);
  trainBtn.addEventListener('click', toggleTraining);
  epochsSlider.addEventListener('input', () => {
    let rawValue = parseInt(epochsSlider.value);
    
    // Custom logic to handle the values
    if (rawValue === 1) {
        epochsPerFrame = 1;
    } else {
        // Round to the nearest 10, but ensure it's at least 10
        epochsPerFrame = Math.max(10, Math.round(rawValue / 10) * 10);
    }
    
    epochsVal.textContent = epochsPerFrame;
});
  addLayerBtn.addEventListener('click', addHiddenLayerControls);
  updateArchBtn.addEventListener('click', updateArchitecture);
  testDataSelect.addEventListener('change', (event) => {
    currentInputIndex = event.target.value;
    updateNodeInfoPanel({ type: 'none' });
  });

  // We no longer need a window resize handler for the canvas
  // window.addEventListener('resize', handleResize);

  populateDatasetSelect();
  datasetSelect.value = currentDatasetName;

  const defaultLayers = datasets[currentDatasetName].network.hiddenLayers;
  const defaultActivations = datasets[currentDatasetName].network.options.activationFunctions;
  defaultLayers.forEach((numNodes, index) => addHiddenLayerControls(numNodes, defaultActivations[index]));
  addOutputLayerControls(defaultActivations[defaultActivations.length - 1]);
  setupNetwork(currentDatasetName);
  populateTestDataSelect();
}

// The handleResize function has been removed.

function draw() {
  background(15, 23, 42);

  if (isTraining) {
    let totalError = 0;
    for (let i = 0; i < epochsPerFrame; i++) {
      let data = random(datasets[currentDatasetName].data);
      nn.train(data.inputs, data.targets);
    }

    for (let data of datasets[currentDatasetName].data) {
      let outputs = nn.predict(data.inputs);
      totalError += nn.calculateError(outputs, data.targets);
    }

    if (totalError / datasets[currentDatasetName].data.length < ERROR_THRESHOLD) {
      stopTraining('Training Complete! Network has learned the pattern.');
    }
  }

  const data = datasets[currentDatasetName].data[currentInputIndex];
  const outputs = nn.predict(data.inputs);
  nnv.show(data.inputs, outputs, currentInputIndex);
  updateDataPanel(data.inputs, outputs, data.targets);
}

function populateDatasetSelect() {
  datasetSelect.innerHTML = '';
  for (const key in datasets) {
    if (datasets.hasOwnProperty(key)) {
      const option = document.createElement('option');
      option.value = key;
      option.textContent = key;
      datasetSelect.appendChild(option);
    }
  }
}

function stopTraining(message) {
  isTraining = false;
  buttonText.textContent = 'Start Training';
  playIcon.classList.remove('hidden');
  pauseIcon.classList.add('hidden');
  trainBtn.classList.remove('danger-button');
  trainBtn.classList.add('control-button');
  trainingStatusElem.textContent = 'Ready';
  trainingStatusElem.className = 'text-sm font-semibold text-green-400';
  datasetSelect.disabled = false;
  updateArchBtn.disabled = false;
  addLayerBtn.disabled = false;
  document.querySelectorAll('.remove-layer-btn').forEach(btn => btn.disabled = false);
  testDataSelect.disabled = false;
  displayTrainingMessage(message, 'success');
}

function displayTrainingMessage(message, type = 'info') {
  const messageDiv = document.createElement('div');
  messageDiv.className = `slide-in rounded-lg p-3 text-sm font-medium ${type === 'success' ? 'bg-green-500/20 text-green-400 border border-green-500/30' :
    type === 'error' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
      'bg-blue-500/20 text-blue-400 border border-blue-500/30'
    }`;
  messageDiv.textContent = message;
  trainingMessageElem.innerHTML = '';
  trainingMessageElem.appendChild(messageDiv);

  setTimeout(() => {
    messageDiv.remove();
  }, 3000);
}

function setupNetwork(datasetName) {
  const config = datasets[datasetName].network;
  nn = new NeuralNetwork(config.inputNodes, config.hiddenLayers, config.outputNodes, {
    ...config.options,
    learning_rate: config.options.learning_rate || 0.01
  });

  // NNvisual now uses the fixed WIDTH and HEIGHT
  nnv = new NNvisual(WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT, nn, { drawmode: 'center', weightThreshold: WEIGHT_THRESHOLD });
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
    displayTrainingMessage('Please pause training before changing the dataset.', 'error');
  }
}

function populateTestDataSelect() {
  testDataSelect.innerHTML = '';
  const currentData = datasets[currentDatasetName].data;
  currentData.forEach((data, index) => {
    const option = document.createElement('option');
    option.value = index;
    option.textContent = `Test Case ${index + 1}: [${data.inputs.join(', ')}]`;
    testDataSelect.appendChild(option);
  });
}

function toggleTraining() {
  isTraining = !isTraining;
  if (isTraining) {
    buttonText.textContent = 'Pause Training';
    playIcon.classList.add('hidden');
    pauseIcon.classList.remove('hidden');
    trainBtn.classList.remove('control-button');
    trainBtn.classList.add('danger-button');
    trainingStatusElem.textContent = 'Training';
    trainingStatusElem.className = 'text-sm font-semibold text-orange-400';
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
  select.className = "custom-select w-full px-2 py-1 bg-neural-700/50 border border-neural-600/50 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-neural-100 text-xs md:text-sm";

  activationFunctionNames.forEach(name => {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name.charAt(0).toUpperCase() + name.slice(1);
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

  const label = document.createElement('label');
  label.className = 'text-neural-400 text-xs md:text-sm font-medium';
  label.textContent = `Layer ${layerCount}:`;

  const input = document.createElement('input');
  input.type = 'number';
  input.value = defaultNodes;
  input.min = '1';
  input.max = '20';
  input.className = 'px-2 py-1 bg-neural-700/50 border border-neural-600/50 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-neural-100 text-xs md:text-sm';

  const removeBtn = document.createElement('button');
  removeBtn.className = 'remove-layer-btn';
  removeBtn.innerHTML = '×';
  removeBtn.title = 'Remove layer';

  div.appendChild(label);
  div.appendChild(input);
  div.appendChild(removeBtn);
  hiddenLayersContainer.appendChild(div);

  const selectGroup = document.createElement('div');
  selectGroup.className = 'space-y-1';
  const selectLabel = document.createElement('label');
  selectLabel.className = "text-neural-400 text-xs font-medium";
  selectLabel.textContent = `Layer ${layerCount} Activation:`;
  const select = createActivationFunctionSelect('hidden', defaultActivation);
  selectGroup.appendChild(selectLabel);
  selectGroup.appendChild(select);

  const outputLayerControl = document.getElementById('output-layer-activation');
  activationFunctionsContainer.insertBefore(selectGroup, outputLayerControl);

  removeBtn.addEventListener('click', () => {
    if (!isTraining) {
      div.remove();
      selectGroup.remove();
      updateLayerLabels();
    } else {
      displayTrainingMessage('Cannot remove a layer while training.', 'error');
    }
  });
}

function addOutputLayerControls(defaultActivation = 'sigmoid') {
  const selectGroup = document.createElement('div');
  selectGroup.id = 'output-layer-activation';
  selectGroup.classList.add('space-y-1');
  const selectLabel = document.createElement('label');
  selectLabel.className = "text-neural-400 text-xs font-medium";
  selectLabel.textContent = `Output Layer Activation:`;
  const select = createActivationFunctionSelect('output', defaultActivation);
  selectGroup.appendChild(selectLabel);
  selectGroup.appendChild(select);
  activationFunctionsContainer.appendChild(selectGroup);
}

function updateLayerLabels() {
  const nodeLabels = hiddenLayersContainer.querySelectorAll('label');
  nodeLabels.forEach((label, index) => {
    label.textContent = `Layer ${index + 1}:`;
  });
  const activationLabels = activationFunctionsContainer.querySelectorAll('label');
  for (let i = 0; i < activationLabels.length; i++) {
    if (activationLabels[i].textContent.startsWith('Layer')) {
      activationLabels[i].textContent = `Layer ${i + 1} Activation:`;
    }
  }
}

function updateArchitecture() {
  if (isTraining) {
    displayTrainingMessage('Cannot update network while training is in progress.', 'error');
    return;
  }

  const newHiddenLayers = [];
  const layerInputs = hiddenLayersContainer.querySelectorAll('input[type="number"]');

  if (layerInputs.length === 0) {
    displayTrainingMessage('You must have at least one hidden layer.', 'error');
    return;
  }

  for (const input of layerInputs) {
    let parsed = parseInt(input.value);
    if (isNaN(parsed) || parsed < 1 || parsed > 20) {
      parsed = Math.max(1, Math.min(20, parsed || 8));
      input.value = parsed;
      displayTrainingMessage('Node count was auto-adjusted to be between 1 and 20.', 'info');
    }
    newHiddenLayers.push(parsed);
  }

  const newActivationFunctions = [];
  const activationSelects = activationFunctionsContainer.querySelectorAll('select');
  for (let i = 0; i < layerInputs.length; i++) {
    newActivationFunctions.push(activationSelects[i].value);
  }
  newActivationFunctions.push(activationSelects[activationSelects.length - 1].value);

  const currentDataset = datasets[currentDatasetName];
  const inputCount = currentDataset.network.inputNodes;
  const outputCount = currentDataset.network.outputNodes;
  const options = {
    ...currentDataset.network.options,
    activationFunctions: newActivationFunctions
  };

  nn = new NeuralNetwork(inputCount, newHiddenLayers, outputCount, options);

  // Re-instantiate NNvisual with the fixed WIDTH and HEIGHT
  nnv = new NNvisual(WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT, nn, { drawmode: 'center', weightThreshold: WEIGHT_THRESHOLD });

  displayTrainingMessage('Network architecture updated successfully!', 'success');
}

function mousePressed() {
  if (mouseX > 0 && mouseX < width && mouseY > 0 && mouseY < height) {
    const nodeInfo = nnv.mousePressed();
    updateNodeInfoPanel(nodeInfo);
  }
}

function updateDataPanel(inputs, outputs, targets) {
  inputValElem.textContent = `[${inputs.map(x => x.toFixed(1)).join(', ')}]`;
  predictedOutputValElem.textContent = `[${outputs.map(x => x.toFixed(3)).join(', ')}]`;
  targetOutputValElem.textContent = `[${targets.map(x => x.toFixed(1)).join(', ')}]`;
}

function updateNodeInfoPanel(nodeInfo) {
  if (nodeInfo.type === 'none') {
    selectedNodeInfoElem.innerHTML = `
                    <div class="flex flex-col items-center justify-center py-6 md:py-8 text-center">
                        <svg class="w-8 md:w-12 h-8 md:h-12 text-neural-500 mb-2 md:mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122"/>
                        </svg>
                        <p class="text-neural-400 text-xs md:text-sm">Click on any node to view detailed information</p>
                    </div>
                `;
  } else if (nodeInfo.type === 'node') {
    let html = `
                    <div class="space-y-2">
                        <div class="bg-neural-700/30 rounded-lg p-2">
                            <h4 class="font-semibold text-blue-400 mb-1 text-sm">Node Info</h4>
                            <div class="grid grid-cols-2 gap-1 text-xs">
                                <div><span class="text-neural-400">Type:</span></div>
                                <div class="text-neural-100">${nodeInfo.layerLabel}</div>
                                <div><span class="text-neural-400">Index:</span></div>
                                <div class="text-neural-100">${nodeInfo.index}</div>
                                <div><span class="text-neural-400">Value:</span></div>
                                <div class="text-green-400 font-mono">${nodeInfo.activationValue?.toFixed(3) || 'N/A'}</div>
                            </div>
                        </div>
                `;

    if (nodeInfo.layer > 0) {
      const weights = nodeInfo.inputDetails?.map(d => parseFloat(d.weight)) || [];
      const avgWeight = weights.length > 0 ? (weights.reduce((a, b) => a + b, 0) / weights.length).toFixed(3) : 'N/A';
      const maxWeight = weights.length > 0 ? Math.max(...weights).toFixed(3) : 'N/A';
      const minWeight = weights.length > 0 ? Math.min(...weights).toFixed(3) : 'N/A';
      const strongConnections = weights.filter(w => Math.abs(w) > 0.5).length;

      html += `
                        <div class="bg-neural-700/30 rounded-lg p-2">
                            <h4 class="font-semibold text-purple-400 mb-1 text-sm">Computation</h4>
                            <div class="grid grid-cols-2 gap-1 text-xs">
                                <div><span class="text-neural-400">Function:</span></div>
                                <div class="text-neutral-100 font-mono">${nodeInfo.activationFunctionName}</div>
                                <div><span class="text-neural-400">Sum:</span></div>
                                <div class="text-orange-400 font-mono">${nodeInfo.weightedSum || 'N/A'}</div>
                                <div><span class="text-neural-400">Bias:</span></div>
                                <div class="text-pink-400 font-mono">${nodeInfo.biasValue || 'N/A'}</div>
                            </div>
                        </div>

                        <div class="bg-neural-700/30 rounded-lg p-2">
                            <div class="flex items-center justify-between mb-1">
                                <h4 class="font-semibold text-cyan-400 text-sm">Weight Stats</h4>
                                <button onclick="toggleWeightDetails()" class="text-xs text-neural-400 hover:text-cyan-400 transition-colors">
                                    <span id="weight-toggle-text">Show All</span>
                                </button>
                            </div>
                            <div class="grid grid-cols-2 gap-1 text-xs mb-2">
                                <div><span class="text-neural-400">Inputs:</span></div>
                                <div class="text-neutral-100">${weights.length}</div>
                                <div><span class="text-neural-400">Strong:</span></div>
                                <div class="text-yellow-400">${strongConnections}</div>
                                <div><span class="text-neural-400">Avg:</span></div>
                                <div class="text-neutral-100 font-mono">${avgWeight}</div>
                                <div><span class="text-neural-400">Range:</span></div>
                                <div class="text-neutral-100 font-mono">${minWeight} to ${maxWeight}</div>
                            </div>
                            
                            <div id="weight-details" class="hidden">
                                <div class="border-t border-neural-600 pt-1 mt-1 max-h-24 overflow-y-auto">
                    `;

      if (nodeInfo.inputDetails && nodeInfo.inputDetails.length > 0) {
        const sortedWeights = nodeInfo.inputDetails
          .map((detail, i) => ({ ...detail, index: i, absWeight: Math.abs(parseFloat(detail.weight)) }))
          .sort((a, b) => b.absWeight - a.absWeight);

        const topWeights = sortedWeights.slice(0, Math.min(5, sortedWeights.length));

        topWeights.forEach((detail) => {
          const isStrong = detail.absWeight > 0.5;
          html += `
                                    <div class="flex justify-between text-xs py-0.5 ${isStrong ? 'text-yellow-400' : ''}">
                                        <span class="text-neural-400">In${detail.index}:</span>
                                        <span class="font-mono">${detail.input} × ${detail.weight}</span>
                                    </div>
                                `;
        });

        if (sortedWeights.length > 5) {
          html += `<div class="text-xs text-neural-500 text-center pt-1">... ${sortedWeights.length - 5} more</div>`;
        }
      }

      html += `
                                </div>
                            </div>
                        </div>
                    `;
    }

    html += `</div>`;
    selectedNodeInfoElem.innerHTML = html;
  } else if (nodeInfo.type === 'bias') {
    let html = `
            <div class="space-y-2">
                <div class="bg-neural-700/30 rounded-lg p-2">
                    <h4 class="font-semibold text-yellow-400 mb-1 text-sm">Bias Node</h4>
                    <div class="grid grid-cols-2 gap-1 text-xs">
                        <div><span class="text-neural-400">Value:</span></div>
                        <div class="text-neutral-100 font-mono">${nodeInfo.biasValue}</div>
                        <div><span class="text-neural-400">Layer:</span></div>
                        <div class="text-neutral-100">${nodeInfo.layer}</div>
                    </div>
                </div>
        `;
    if (nodeInfo.biasWeights && nodeInfo.biasWeights.length > 0) {
      html += `<div class="bg-neural-700/30 rounded-lg p-2">
                <h4 class="font-semibold text-orange-400 mb-1 text-sm">Connections (${nodeInfo.biasWeights.length})</h4>
                <div class="max-h-20 overflow-y-auto space-y-0.5">
            `;
      nodeInfo.biasWeights.forEach((detail) => {
        const isStrong = Math.abs(parseFloat(detail.weight)) > 0.5;
        html += `
                    <div class="flex justify-between text-xs ${isStrong ? 'text-yellow-400' : ''}">
                        <span class="text-neural-400">Node ${detail.node}:</span>
                        <span class="font-mono">${detail.weight}</span>
                    </div>
                `;
      });
      html += `</div></div>`;
    }
    html += `</div>`;
    selectedNodeInfoElem.innerHTML = html;
  }
}

// Toggle function for weight details
window.toggleWeightDetails = function () {
  const details = document.getElementById('weight-details');
  const toggleText = document.getElementById('weight-toggle-text');
  if (details && toggleText) {
    if (details.classList.contains('hidden')) {
      details.classList.remove('hidden');
      toggleText.textContent = 'Hide';
    } else {
      details.classList.add('hidden');
      toggleText.textContent = 'Show All';
    }
  }
}
