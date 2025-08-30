class NNvisual {
    /**
     * @param {number} x_ - The x-coordinate of the bounding box.
     * @param {number} y_ - The y-coordinate of the bounding box.
     * @param {number} w_ - The width of the bounding box.
     * @param {number} h_ - The height of the bounding box.
     * @param {NeuralNetwork} nn_ - The neural network object to visualize.
     * @param {Object} options - Visualization options.
     */
    constructor(x_, y_, w_, h_, nn_, options = { drawmode: 'center', showinfobox: true }) {
        this.options = options;
        this.drawmode = options.drawmode.toLowerCase() || 'default';

        if (this.drawmode === "center") {
            this.x = x_ - w_ / 2;
            this.y = y_ - h_ / 2;
        } else {
            this.x = x_;
            this.y = y_;
        }

        this.w = w_;
        this.h = h_;
        this.nn = nn_;
        this.showinfobox = options.showinfobox !== undefined ? options.showinfobox : true;

        this.initializeSizing();
        this.selectedNode = null;
        this.activations = null;
        this.nodePositions = this.calculateLayerPositions();
        this.biasNodePositions = this.calculateBiasPositions();

        // Define colors here to be accessible throughout the class
        this.CONNECTION_POSITIVE_COLOR = color(150, 250, 150, 150);
        this.CONNECTION_NEGATIVE_COLOR = color(250, 150, 150, 150);
        this.CONNECTION_NEUTRAL_COLOR = color(50, 50, 50, 80);
        this.SLOW_PARTICLE_COLOR = color(100, 100, 255, 180);
        this.FAST_PARTICLE_COLOR = color(255, 100, 100, 200);
        this.NEUTRAL_PARTICLE_COLOR = color(255, 255, 255, 150);
        this.NEUTRAL_COLOR = color(150, 150, 150);
        this.HEATMAP_POSITIVE_COLOR = color(255, 100, 100);
        this.HEATMAP_NEGATIVE_COLOR = color(100, 100, 255);
        this.TEXT_COLOR = color(220);
    }

    /**
     * Initializes dynamic sizing parameters for the visualization.
     */
    initializeSizing() {
        const hiddenLayersArray = Array.isArray(this.nn.hidden_layers) ? this.nn.hidden_layers : [this.nn.hidden_layers];
        const layerSizes = [this.nn.input_nodes, ...hiddenLayersArray, this.nn.output_nodes];
        const maxNodesInLayer = Math.max(...layerSizes);
        const maxRadius = 30;
        this.r = min(maxRadius, this.h / (maxNodesInLayer * 4));
        this.NODE_HEIGHT = this.r * 1.5;
        this.NODE_GAP = this.r * 1.5;
    }

    /**
     * Calculates the position of each node.
     * @returns {Array<Array<p5.Vector>>} An array of node positions.
     */
    calculateLayerPositions() {
        let nodePositions = [];
        const hiddenLayersArray = Array.isArray(this.nn.hidden_layers) ? this.nn.hidden_layers : [this.nn.hidden_layers];
        const layerSizes = [this.nn.input_nodes, ...hiddenLayersArray, this.nn.output_nodes];
        const numLayers = layerSizes.length;
        const layerXs = this.calculateLayerXPositions(numLayers);

        // Calculate the maximum height required for all layers (not including bias nodes yet)
        let maxLayerHeight = 0;
        for (let l = 0; l < numLayers; l++) {
            const numNodes = layerSizes[l];
            let layerHeight = 0;
            if (numNodes > 20) {
                const numRows = ceil(numNodes / ceil(sqrt(numNodes)));
                layerHeight = (numRows * this.NODE_HEIGHT) + ((numRows - 1) * this.NODE_GAP);
            } else {
                layerHeight = (numNodes * this.NODE_HEIGHT) + ((numNodes - 1) * this.NODE_GAP);
            }
            maxLayerHeight = max(maxLayerHeight, layerHeight);
        }

        // Calculate the total height needed for both the network and the biases
        const biasVerticalSpace = (this.NODE_HEIGHT + this.NODE_GAP) * (numLayers > 1 ? 1 : 0);
        const totalHeightWithBias = maxLayerHeight + biasVerticalSpace;

        // Adjust the overall start position to center the entire network including biases
        const startY = this.y + (this.h - totalHeightWithBias) / 2;

        for (let l = 0; l < numLayers; l++) {
            const numNodes = layerSizes[l];
            const layerX = layerXs[l];
            const currentLayerPositions = [];
            const GRID_THRESHOLD = 20;

            if (numNodes > GRID_THRESHOLD) {
                this.calculateGridPositions(numNodes, layerX, currentLayerPositions, startY, maxLayerHeight);
            } else {
                this.calculateColumnPositions(numNodes, layerX, currentLayerPositions, startY, maxLayerHeight);
            }
            nodePositions.push(currentLayerPositions);
        }
        return nodePositions;
    }

    /**
     * Calculates the horizontal positions for each layer.
     * @param {number} numLayers - The number of layers.
     * @returns {Array<number>} An array of x-coordinates for each layer.
     */
    calculateLayerXPositions(numLayers) {
        let layerXs = [];
        const layerPadding = this.r * 3;
        const totalHorizontalSpace = this.w - layerPadding * 2;
        const numGaps = numLayers - 1;

        for (let l = 0; l < numLayers; l++) {
            const layerX = this.x + layerPadding + (l / max(1, numGaps)) * totalHorizontalSpace;
            layerXs.push(layerX);
        }
        return layerXs;
    }

    /**
     * Calculates positions for nodes in a grid layout.
     * @param {number} numNodes - The number of nodes in the layer.
     * @param {number} layerX - The horizontal position of the layer.
     * @param {Array<p5.Vector>} positionsArray - The array to store calculated positions.
     * @param {number} startY - The vertical start position for the whole network.
     * @param {number} maxLayerHeight - The maximum height of any layer.
     */
    calculateGridPositions(numNodes, layerX, positionsArray, startY, maxLayerHeight) {
        const numCols = ceil(sqrt(numNodes));
        const numRows = ceil(numNodes / numCols);
        const totalGridWidth = (numCols * this.NODE_HEIGHT) + ((numCols - 1) * this.NODE_GAP);
        const totalGridHeight = (numRows * this.NODE_HEIGHT) + ((numRows - 1) * this.NODE_GAP);

        // Center the grid vertically within the max layer height
        const gridStartY = startY + (maxLayerHeight - totalGridHeight) / 2;
        const gridStartX = layerX - totalGridWidth / 2;

        for (let i = 0; i < numNodes; i++) {
            const row = floor(i / numCols);
            const col = i % numCols;
            const nodeX = gridStartX + col * (this.NODE_HEIGHT + this.NODE_GAP);
            const nodeY = gridStartY + row * (this.NODE_HEIGHT + this.NODE_GAP);
            positionsArray.push(createVector(nodeX, nodeY));
        }
    }

    /**
     * Calculates positions for nodes in a single column layout.
     * @param {number} numNodes - The number of nodes in the layer.
     * @param {number} layerX - The horizontal position of the layer.
     * @param {Array<p5.Vector>} positionsArray - The array to store calculated positions.
     * @param {number} startY - The vertical start position for the whole network.
     * @param {number} maxLayerHeight - The maximum height of any layer.
     */
    calculateColumnPositions(numNodes, layerX, positionsArray, startY, maxLayerHeight) {
        const totalRequiredHeight = (numNodes * this.NODE_HEIGHT) + ((numNodes - 1) * this.NODE_GAP);

        // Center the column vertically within the max layer height
        const columnStartY = startY + (maxLayerHeight - totalRequiredHeight) / 2;

        for (let i = 0; i < numNodes; i++) {
            positionsArray.push(createVector(layerX, columnStartY + i * (this.NODE_HEIGHT + this.NODE_GAP)));
        }
    }

    /**
     * Calculates the position of the bias nodes.
     * @returns {Array<p5.Vector>} An array of bias node positions.
     */
    calculateBiasPositions() {
        let biasPositions = [];
        const numLayers = this.nodePositions.length;
        let maxYOfAllNodes = 0;

        for (const layerPositions of this.nodePositions) {
            for (const pos of layerPositions) {
                maxYOfAllNodes = max(maxYOfAllNodes, pos.y);
            }
        }
        const commonBiasY = maxYOfAllNodes + this.NODE_HEIGHT + this.NODE_GAP;

        for (let l = 0; l < numLayers - 1; l++) {
            const prevLayerPositions = this.nodePositions[l];
            const biasX = prevLayerPositions[0].x;
            biasPositions.push(createVector(biasX, commonBiasY));
        }
        return biasPositions;
    }

    show(inputs, outputs, currentInputIndex = null) {
        this.drawBoundingBox();
        if (typeof this.nn.feedForwardAllLayers !== 'function') {
            console.error("Error: The provided 'nn' object does not have the 'feedForwardAllLayers' method.");
            return;
        }

        this.activations = this.nn.feedForwardAllLayers(inputs);

        this.drawConnections(currentInputIndex);
        this.drawNodes();
    }

    drawBoundingBox() {
        noFill();
        stroke(255, 20);
        strokeWeight(2);
        rect(this.x, this.y, this.w, this.h);
    }

    drawConnections(currentInputIndex) {
        const numLayers = this.nodePositions.length;
        // Check if the selected node is a regular node and it's in the output layer
        if (this.selectedNode && this.selectedNode.type === 'node' && this.selectedNode.layer === numLayers - 1) {
            this.drawSelectedPathConnections();
        } else {
            this.drawAllConnections();
        }
    }

    drawSelectedPathConnections() {
        const numLayers = this.nodePositions.length;
        const WEIGHT_THRESHOLD = 0.8;
        let nodesInPath = new Set();
        let inputs_ = this.nn.lastInputs;

        // Start with the selected node
        nodesInPath.add(`${this.selectedNode.layer},${this.selectedNode.index}`);

        // Trace back the path from the selected node
        for (let l = numLayers - 1; l > 0; l--) {
            let prevLayerNodes = this.nodePositions[l - 1];
            let nextLayerNodes = this.nodePositions[l];
            let weightsMatrix = this.nn.weights[l - 1];
            let biasConnectionsMatrix = this.nn.biases[l - 1];
            let newNodesInPrevLayer = new Set();

            for (let j = 0; j < nextLayerNodes.length; j++) {
                if (nodesInPath.has(`${l},${j}`)) {
                    const biasWeight = biasConnectionsMatrix.data[j][0];
                    if (abs(biasWeight) > WEIGHT_THRESHOLD) {
                        this.drawConnectionAndParticle(this.biasNodePositions[l - 1], nextLayerNodes[j], biasWeight, 1);
                    }

                    for (let i = 0; i < prevLayerNodes.length; i++) {
                        const weight = weightsMatrix.data[j][i];
                        const activationStrength = (this.activations[l - 1] && this.activations[l - 1].data) ? this.activations[l - 1].data[i][0] : 0;

                        // Check if the current node is an input node with a non-zero value
                        const isInputNode = (l - 1 === 0);
                        const isInputActive = isInputNode && inputs_[i] !== 0;

                        // The logic change is here:
                        // Draw connections if the weight is significant OR the source node is an active input node.
                        if (abs(weight) > WEIGHT_THRESHOLD || isInputActive) {
                            this.drawConnectionAndParticle(prevLayerNodes[i], nextLayerNodes[j], weight, activationStrength);
                            newNodesInPrevLayer.add(`${l - 1},${i}`);
                        }
                    }
                }
            }
            nodesInPath = newNodesInPrevLayer;
        }
    }

    drawAllConnections() {
        const numLayers = this.nodePositions.length;
        for (let l = 0; l < numLayers - 1; l++) {
            const currentLayerNodes = this.nodePositions[l];
            const nextLayerNodes = this.nodePositions[l + 1];

            const biasConnectionsMatrix = this.nn.biases[l];
            for (let j = 0; j < nextLayerNodes.length; j++) {
                const biasWeight = biasConnectionsMatrix.data[j][0];
                this.drawConnectionAndParticle(this.biasNodePositions[l], nextLayerNodes[j], biasWeight, 1);
            }

            const weightsMatrix = this.nn.weights[l];
            for (let i = 0; i < currentLayerNodes.length; i++) {
                for (let j = 0; j < nextLayerNodes.length; j++) {
                    const weight = weightsMatrix.data[j][i];
                    const activationStrength = (this.activations[l] && this.activations[l].data) ? this.activations[l].data[i][0] : 0;
                    this.drawConnectionAndParticle(currentLayerNodes[i], nextLayerNodes[j], weight, activationStrength);
                }
            }
        }
    }

    drawConnectionAndParticle(startPos, endPos, weight, activationStrength) {

        const clampedWeight = constrain(abs(weight), 0, 1);
        const strokeW = map(clampedWeight, 0, 1, 1, 3);
        const connectionColor = weight > 0 ?
            lerpColor(this.CONNECTION_NEUTRAL_COLOR, this.CONNECTION_POSITIVE_COLOR, map(weight, 0, 1, 0, 1)) :
            lerpColor(this.CONNECTION_NEUTRAL_COLOR, this.CONNECTION_NEGATIVE_COLOR, map(abs(weight), 0, 1, 0, 1));

        strokeWeight(strokeW);
        stroke(connectionColor);
        line(startPos.x, startPos.y, endPos.x, endPos.y);

        let direction = p5.Vector.sub(endPos, startPos).normalize();
        let initialParticlePos = p5.Vector.add(startPos, p5.Vector.mult(direction, this.r));

        let distance = dist(initialParticlePos.x, initialParticlePos.y, endPos.x, endPos.y);

        const MIN_SPEED = 0.5;
        const MAX_SPEED = 5;
        const particleSpeed = map(clampedWeight, 0, 1, MIN_SPEED, MAX_SPEED);

        let currentDist = (frameCount * particleSpeed) % distance;

        let animationProgress = currentDist / distance;
        let animX = lerp(initialParticlePos.x, endPos.x, animationProgress);
        let animY = lerp(initialParticlePos.y, endPos.y, animationProgress);

        noStroke();

        const speedNormalized = map(particleSpeed, MIN_SPEED, MAX_SPEED, 0, 1);
        let particleColor;

        if (speedNormalized < 0.5) {
            particleColor = lerpColor(this.NEUTRAL_PARTICLE_COLOR, this.SLOW_PARTICLE_COLOR, map(speedNormalized, 0, 0.5, 0, 1));
        } else {
            particleColor = lerpColor(this.NEUTRAL_PARTICLE_COLOR, this.FAST_PARTICLE_COLOR, map(speedNormalized, 0.5, 1, 0, 1));
        }

        particleColor.setAlpha(map(abs(activationStrength), 0, 1, 50, 255));

        fill(particleColor);
        ellipse(animX, animY, strokeW + 2, strokeW + 2);
    }

    drawNodes() {
        this.drawNeuralNodes();
        this.drawBiasNodes();
    }

    drawNeuralNodes() {
        for (let l = 0; l < this.nodePositions.length; l++) {
            const layerNodes = this.nodePositions[l];
            for (let i = 0; i < layerNodes.length; i++) {
                const pos = layerNodes[i];
                const activationValue = (this.activations[l] && this.activations[l].data) ? this.activations[l].data[i][0] : 0;
                let nodeColor;

                if (activationValue > 0) {
                    nodeColor = lerpColor(this.NEUTRAL_COLOR, this.HEATMAP_POSITIVE_COLOR, map(activationValue, 0, 1, 0, 1));
                } else {
                    nodeColor = lerpColor(this.NEUTRAL_COLOR, this.HEATMAP_NEGATIVE_COLOR, map(abs(activationValue), 0, 1, 0, 1));
                }

                noStroke();
                fill(nodeColor);
                ellipse(pos.x, pos.y, this.r * 2);

                if (this.isSelected(l, i, 'node')) {
                    this.highlightNode(pos);
                }

                fill(this.TEXT_COLOR);
                textSize(this.r * 0.75);
                stroke(0);
                strokeWeight(this.r * 0.2);
                textAlign(CENTER, CENTER);
                text(nf(activationValue, 1, 2), pos.x, pos.y);
            }
        }
    }

    drawBiasNodes() {
        for (let l = 0; l < this.biasNodePositions.length; l++) {
            const pos = this.biasNodePositions[l];
            const biasValue = this.nn.biases[l].data[0][0];

            fill(this.NEUTRAL_COLOR);
            stroke(biasValue > 0 ? color(100, 200, 100) : color(200, 100, 100));
            strokeWeight(3);
            rectMode(CENTER);
            rect(pos.x, pos.y, this.r * 1.5, this.r * 1.5, 5);

            if (this.isSelected(l + 1, 0, 'bias')) {
                this.highlightBiasNode(pos);
            }

            fill(this.TEXT_COLOR);
            noStroke();
            textSize(this.r * 0.75);
            textAlign(CENTER, CENTER);
            stroke(0);
            text("1.00", pos.x, pos.y);
        }
        rectMode(CORNER);
    }

    isSelected(layer, index, type) {
        return this.selectedNode && this.selectedNode.type === type && this.selectedNode.layer === layer && this.selectedNode.index === index;
    }

    highlightNode(pos) {
        noFill();
        stroke(255, 255, 0);
        strokeWeight(3);
        ellipse(pos.x, pos.y, this.r * 2 + 6);
    }

    highlightBiasNode(pos) {
        noFill();
        stroke(255, 255, 0);
        strokeWeight(3);
        rect(pos.x, pos.y, this.r * 1.5 + 6, this.r * 1.5 + 6, 5);
    }

    mousePressed() {
        this.selectedNode = null;
        for (let l = 0; l < this.nodePositions.length; l++) {
            for (let i = 0; i < this.nodePositions[l].length; i++) {
                const pos = this.nodePositions[l][i];
                if (dist(mouseX, mouseY, pos.x, pos.y) < this.r) {
                    this.selectedNode = { layer: l, index: i, type: 'node' };
                    return this.getInfoBoxText(l, i, 'node');
                }
            }
        }
        for (let l = 0; l < this.biasNodePositions.length; l++) {
            const pos = this.biasNodePositions[l];
            if (dist(mouseX, mouseY, pos.x, pos.y) < this.r * 1.5) {
                this.selectedNode = { layer: l + 1, index: 0, type: 'bias' };
                return this.getInfoBoxText(l + 1, 0, 'bias');
            }
        }
        return { type: 'none' };
    }

    getInfoBoxText(layer, index, type) {
        let info = { type: type, layer: layer, index: index };
        let layerLabel = "";
        let inputDetails = [];
        let weightedSum = 0;

        info.layerLabel = this.getLayerLabel(layer);

        if (type === 'node') {
            info.activationValue = (this.activations[layer] && this.activations[layer].data) ? this.activations[layer].data[index][0] : 0;
            info.activationFunctionName = layer > 0 ? (this.nn.activation_functions[layer - 1] ? this.nn.activation_functions[layer - 1].name : 'N/A') : 'N/A (Input)';

            if (layer > 0) {
                let prevActivations = this.activations[layer - 1].data;
                let weightsToThisNode = this.nn.weights[layer - 1].data[index];
                let biasValue = this.nn.biases[layer - 1].data[index][0];

                for (let i = 0; i < weightsToThisNode.length; i++) {
                    let input = prevActivations[i][0];
                    let weight = weightsToThisNode[i];
                    weightedSum += input * weight;
                    inputDetails.push({ input: nf(input, 1, 2), weight: nf(weight, 1, 2) });
                }
                weightedSum += biasValue;

                info.weightedSum = nf(weightedSum, 1, 2);
                info.biasValue = nf(biasValue, 1, 2);
                info.inputDetails = inputDetails;
            }
        } else if (type === 'bias') {
            let nextLayerSize = this.nodePositions[layer].length;
            let biases = this.nn.biases[layer - 1].data;
            let biasWeights = [];
            for (let i = 0; i < nextLayerSize; i++) {
                biasWeights.push({ node: i, weight: nf(biases[i][0], 1, 2) });
            }
            info.biasValue = nf(1.0, 1, 2);
            info.biasWeights = biasWeights;
        }

        return info;
    }

    getLayerLabel(layer) {
        if (layer === 0) return 'Input';
        if (layer === this.nodePositions.length - 1) return 'Output';
        return `Hidden ${layer}`;
    }
}