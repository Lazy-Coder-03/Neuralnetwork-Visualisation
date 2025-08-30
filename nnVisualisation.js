// nnVisualisation.js - Professional Visualization (Refactored)

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
        this.NODE_HEIGHT = this.r * 2;
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

        for (let l = 0; l < numLayers; l++) {
            const numNodes = layerSizes[l];
            const layerX = layerXs[l];
            const currentLayerPositions = [];
            const GRID_THRESHOLD = 20;

            if (numNodes > GRID_THRESHOLD) {
                this.calculateGridPositions(numNodes, layerX, currentLayerPositions);
            } else {
                this.calculateColumnPositions(numNodes, layerX, currentLayerPositions);
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
     */
    calculateGridPositions(numNodes, layerX, positionsArray) {
        const numCols = ceil(sqrt(numNodes));
        const numRows = ceil(numNodes / numCols);
        const totalGridWidth = (numCols * this.NODE_HEIGHT) + ((numCols - 1) * this.NODE_GAP);
        const totalGridHeight = (numRows * this.NODE_HEIGHT) + ((numRows - 1) * this.NODE_GAP);
        const startY = this.y + (this.h - totalGridHeight) / 2;
        const gridStartX = layerX - totalGridWidth / 2;

        for (let i = 0; i < numNodes; i++) {
            const row = floor(i / numCols);
            const col = i % numCols;
            const nodeX = gridStartX + col * (this.NODE_HEIGHT + this.NODE_GAP);
            const nodeY = startY + row * (this.NODE_HEIGHT + this.NODE_GAP);
            positionsArray.push(createVector(nodeX, nodeY));
        }
    }

    /**
     * Calculates positions for nodes in a single column layout.
     * @param {number} numNodes - The number of nodes in the layer.
     * @param {number} layerX - The horizontal position of the layer.
     * @param {Array<p5.Vector>} positionsArray - The array to store calculated positions.
     */
    calculateColumnPositions(numNodes, layerX, positionsArray) {
        const totalRequiredHeight = (numNodes * this.NODE_HEIGHT) + ((numNodes - 1) * this.NODE_GAP);
        const startY = this.y + (this.h - totalRequiredHeight) / 2;

        for (let i = 0; i < numNodes; i++) {
            positionsArray.push(createVector(layerX, startY + i * (this.NODE_HEIGHT + this.NODE_GAP)));
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

        // Find the max Y position of all nodes across all layers
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
        if (this.showinfobox) {
            this.drawInfoBox();
        }
    }

    drawBoundingBox() {
        noFill();
        stroke(255, 20);
        strokeWeight(2);
        rect(this.x, this.y, this.w, this.h);
    }

    drawConnections(currentInputIndex) {
        if (this.selectedNode && this.selectedNode.type === 'node' && this.selectedNode.layer === this.nodePositions.length - 1) {
            // If an output node is selected, draw the path
            if (currentInputIndex !== null) {
                this.drawSelectedPathConnections(currentInputIndex);
            } else {
                // Fallback to the original logic if no input index is provided
                this.drawSelectedNodeConnections();
            }
        } else {
            // If no output node is selected, draw all connections
            this.drawAllConnections();
        }
    }

    /**
     * Draws connections and particles for a selected output node.
     */
    drawSelectedNodeConnections() {
        const numLayers = this.nodePositions.length;
        const WEIGHT_THRESHOLD = 0.5;
        let selectedNodesToDraw = new Set();
        selectedNodesToDraw.add(`${this.selectedNode.layer},${this.selectedNode.index}`);

        for (let l = numLayers - 1; l > 0; l--) {
            const prevLayerNodes = this.nodePositions[l - 1];
            const nextLayerNodes = this.nodePositions[l];
            const weightsMatrix = this.nn.weights[l - 1];
            const biasConnectionsMatrix = this.nn.biases[l - 1];
            let newNodesInPrevLayer = new Set();

            for (let j = 0; j < nextLayerNodes.length; j++) {
                if (selectedNodesToDraw.has(`${l},${j}`)) {
                    // Draw bias connection
                    const biasWeight = biasConnectionsMatrix.data[j][0];
                    if (abs(biasWeight) > WEIGHT_THRESHOLD) {
                        this.drawConnectionAndParticle(this.biasNodePositions[l - 1], nextLayerNodes[j], biasWeight, 1);
                    }

                    // Draw connections from previous nodes
                    for (let i = 0; i < prevLayerNodes.length; i++) {
                        const weight = weightsMatrix.data[j][i];
                        if (abs(weight) > WEIGHT_THRESHOLD) {
                            newNodesInPrevLayer.add(`${l - 1},${i}`);
                            const activationStrength = (this.activations[l - 1] && this.activations[l - 1].data) ? this.activations[l - 1].data[i][0] : 0;
                            this.drawConnectionAndParticle(prevLayerNodes[i], nextLayerNodes[j], weight, activationStrength);
                        }
                    }
                }
            }
            selectedNodesToDraw = newNodesInPrevLayer;
        }
    }
    /**
 * Draws the connections for a single path from the current input to the selected output node.
 * @param {number} currentInputIndex - The index of the active input node.
 */
    drawSelectedPathConnections(currentInputIndex) {
        const numLayers = this.nodePositions.length;
        const WEIGHT_THRESHOLD = 0.5;
        let nodesInPath = new Set();

        // Add the selected output node to the path
        nodesInPath.add(`${numLayers - 1},${this.selectedNode.index}`);

        // Recursively add all connected nodes in previous layers
        for (let l = numLayers - 1; l > 0; l--) {
            let prevLayerNodes = this.nodePositions[l - 1];
            let nextLayerNodes = this.nodePositions[l];
            let weightsMatrix = this.nn.weights[l - 1];
            let biasConnectionsMatrix = this.nn.biases[l - 1];
            let newNodesInPrevLayer = new Set();

            for (let j = 0; j < nextLayerNodes.length; j++) {
                if (nodesInPath.has(`${l},${j}`)) {
                    // Draw bias connection to this node if it meets the threshold
                    const biasWeight = biasConnectionsMatrix.data[j][0];
                    if (abs(biasWeight) > WEIGHT_THRESHOLD) {
                        this.drawConnectionAndParticle(this.biasNodePositions[l - 1], nextLayerNodes[j], biasWeight, 1);
                    }

                    // Draw connections from previous nodes to this node
                    for (let i = 0; i < prevLayerNodes.length; i++) {
                        const weight = weightsMatrix.data[j][i];
                        // Only process connections that meet the weight threshold
                        if (abs(weight) > WEIGHT_THRESHOLD) {
                            // For the first hidden layer, check if the previous node is the active input node
                            if (l === 1 && i !== currentInputIndex) {
                                continue;
                            }

                            const activationStrength = (this.activations[l - 1] && this.activations[l - 1].data) ? this.activations[l - 1].data[i][0] : 0;
                            this.drawConnectionAndParticle(prevLayerNodes[i], nextLayerNodes[j], weight, activationStrength);

                            newNodesInPrevLayer.add(`${l - 1},${i}`);
                        }
                    }
                }
            }
            nodesInPath = newNodesInPrevLayer;
        }
    }

    /**
     * Draws all connections and particles.
     */
    drawAllConnections() {
        const numLayers = this.nodePositions.length;
        for (let l = 0; l < numLayers - 1; l++) {
            const currentLayerNodes = this.nodePositions[l];
            const nextLayerNodes = this.nodePositions[l + 1];

            // Draw bias connections
            const biasConnectionsMatrix = this.nn.biases[l];
            for (let j = 0; j < nextLayerNodes.length; j++) {
                const biasWeight = biasConnectionsMatrix.data[j][0];
                this.drawConnectionAndParticle(this.biasNodePositions[l], nextLayerNodes[j], biasWeight, 1);
            }

            // Draw regular connections
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

    /**
     * Helper function to draw a single connection and its particle.
     * @param {p5.Vector} startPos - The starting position.
     * @param {p5.Vector} endPos - The ending position.
     * @param {number} weight - The weight of the connection.
     * @param {number} activationStrength - The activation strength for particle animation.
     */
    drawConnectionAndParticle(startPos, endPos, weight, activationStrength) {
        const CONNECTION_POSITIVE_COLOR = color(150, 250, 150, 150);
        const CONNECTION_NEGATIVE_COLOR = color(250, 150, 150, 150);
        const CONNECTION_NEUTRAL_COLOR = color(50, 50, 50, 80);

        const clampedWeight = constrain(abs(weight), 0, 1);
        const strokeW = map(clampedWeight, 0, 1, 1, 3);
        const connectionColor = weight > 0 ?
            lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_POSITIVE_COLOR, map(weight, 0, 1, 0, 1)) :
            lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_NEGATIVE_COLOR, map(abs(weight), 0, 1, 0, 1));

        strokeWeight(strokeW);
        stroke(connectionColor);
        line(startPos.x, startPos.y, endPos.x, endPos.y);

        // Draw particle
        let direction = p5.Vector.sub(endPos, startPos).normalize();
        let initialParticlePos = p5.Vector.add(startPos, p5.Vector.mult(direction, this.r));

        let distance = p5.Vector.dist(initialParticlePos, endPos);

        // Map the clamped weight to a speed range (e.g., 0.1 to 10 pixels/frame)
        const MIN_SPEED = 0.1;
        const MAX_SPEED = 10.0;
        const particleSpeed = map(clampedWeight, 0, 1, MIN_SPEED, MAX_SPEED);

        // Use the calculated speed for the animation
        let currentDist = (frameCount * particleSpeed) % distance;

        // Recalculate animation progress based on distance
        let animationProgress = currentDist / distance;

        let animX = lerp(initialParticlePos.x, endPos.x, animationProgress);
        let animY = lerp(initialParticlePos.y, endPos.y, animationProgress);

        noStroke();

        // ✨ BEGIN CHANGE for particle color based on speed ✨
        // Define a color gradient for the particle based on speed
        const SLOW_PARTICLE_COLOR = color(100, 100, 255, 180); // Blueish for slow
        const FAST_PARTICLE_COLOR = color(255, 100, 100, 200); // Reddish for fast
        const NEUTRAL_PARTICLE_COLOR = color(255, 255, 255, 150); // White for moderate

        // Determine particle color based on its speed, relative to the min/max speed
        // We map the actual particleSpeed (from MIN_SPEED to MAX_SPEED) to a 0-1 range
        const speedNormalized = map(particleSpeed, MIN_SPEED, MAX_SPEED, 0, 1);
        let particleColor;

        if (speedNormalized < 0.5) {
            // Interpolate between neutral and slow color for lower speeds
            particleColor = lerpColor(NEUTRAL_PARTICLE_COLOR, SLOW_PARTICLE_COLOR, map(speedNormalized, 0, 0.5, 0, 1));
        } else {
            // Interpolate between neutral and fast color for higher speeds
            particleColor = lerpColor(NEUTRAL_PARTICLE_COLOR, FAST_PARTICLE_COLOR, map(speedNormalized, 0.5, 1, 0, 1));
        }

        // You can also consider the activation strength for an additional layer of color information
        // For simplicity, let's primarily use speed for the base color, and activation for intensity.
        // The previous `abs(activationStrength)` was already used for alpha, which is good.
        // Let's blend the base particleColor with a transparency based on activation strength
        particleColor.setAlpha(map(abs(activationStrength), 0, 1, 50, 255)); // More activated, more opaque

        fill(particleColor);
        // ✨ END CHANGE ✨

        ellipse(animX, animY, strokeW + 2, strokeW + 2);
    }

    drawNodes() {
        this.drawNeuralNodes();
        this.drawBiasNodes();
    }

    /**
     * Draws the main neural network nodes.
     */
    drawNeuralNodes() {
        const NEUTRAL_COLOR = color(150, 150, 150);
        const HEATMAP_POSITIVE_COLOR = color(255, 100, 100);
        const HEATMAP_NEGATIVE_COLOR = color(100, 100, 255);
        const TEXT_COLOR = color(220);

        for (let l = 0; l < this.nodePositions.length; l++) {
            const layerNodes = this.nodePositions[l];
            for (let i = 0; i < layerNodes.length; i++) {
                const pos = layerNodes[i];
                const activationValue = (this.activations[l] && this.activations[l].data) ? this.activations[l].data[i][0] : 0;
                let nodeColor;

                if (activationValue > 0) {
                    nodeColor = lerpColor(NEUTRAL_COLOR, HEATMAP_POSITIVE_COLOR, map(activationValue, 0, 1, 0, 1));
                } else {
                    nodeColor = lerpColor(NEUTRAL_COLOR, HEATMAP_NEGATIVE_COLOR, map(abs(activationValue), 0, 1, 0, 1));
                }

                noStroke();
                fill(nodeColor);
                ellipse(pos.x, pos.y, this.r * 2);

                if (this.isSelected(l, i, 'node')) {
                    this.highlightNode(pos);
                }

                fill(TEXT_COLOR);
                textSize(this.r * 0.75);
                stroke(0);
                strokeWeight(this.r * 0.2);
                textAlign(CENTER, CENTER);
                text(nf(activationValue, 1, 2), pos.x, pos.y);
            }
        }
    }

    /**
     * Draws the bias nodes.
     */
    drawBiasNodes() {
        const NEUTRAL_COLOR = color(150, 150, 150);
        const TEXT_COLOR = color(220);

        for (let l = 0; l < this.biasNodePositions.length; l++) {
            const pos = this.biasNodePositions[l];
            const biasValue = this.nn.biases[l].data[0][0];

            fill(NEUTRAL_COLOR);
            stroke(biasValue > 0 ? color(100, 200, 100) : color(200, 100, 100));
            strokeWeight(3);
            rectMode(CENTER);
            rect(pos.x, pos.y, this.r * 1.5, this.r * 1.5, 5);

            if (this.isSelected(l + 1, 0, 'bias')) {
                this.highlightBiasNode(pos);
            }

            fill(TEXT_COLOR);
            noStroke();
            textSize(this.r * 0.75);
            textAlign(CENTER, CENTER);
            stroke(0);
            text("1.00", pos.x, pos.y);
        }
        rectMode(CORNER);
    }

    /**
     * Checks if a given node or bias is currently selected.
     * @param {number} layer - The layer index.
     * @param {number} index - The node index.
     * @param {string} type - 'node' or 'bias'.
     * @returns {boolean} True if selected, otherwise false.
     */
    isSelected(layer, index, type) {
        return this.selectedNode && this.selectedNode.type === type && this.selectedNode.layer === layer && this.selectedNode.index === index;
    }

    /**
     * Draws a highlight around a selected node.
     * @param {p5.Vector} pos - The position of the node.
     */
    highlightNode(pos) {
        noFill();
        stroke(255, 255, 0); // Yellow highlight
        strokeWeight(3);
        ellipse(pos.x, pos.y, this.r * 2 + 6);
    }

    /**
     * Draws a highlight around a selected bias node.
     * @param {p5.Vector} pos - The position of the bias node.
     */
    highlightBiasNode(pos) {
        noFill();
        stroke(255, 255, 0); // Yellow highlight
        strokeWeight(3);
        rect(pos.x, pos.y, this.r * 1.5 + 6, this.r * 1.5 + 6, 5);
    }

    /**
     * Handles mouse press events to select a node or bias.
     */
    mousePressed() {
        this.selectedNode = null;
        for (let l = 0; l < this.nodePositions.length; l++) {
            for (let i = 0; i < this.nodePositions[l].length; i++) {
                const pos = this.nodePositions[l][i];
                if (dist(mouseX, mouseY, pos.x, pos.y) < this.r) {
                    this.selectedNode = { layer: l, index: i, type: 'node' };
                    console.log(`Selected Node - Layer: ${l}, Index: ${i}`);
                    return;
                }
            }
        }
        for (let l = 0; l < this.biasNodePositions.length; l++) {
            const pos = this.biasNodePositions[l];
            if (dist(mouseX, mouseY, pos.x, pos.y) < this.r * 1.5) {
                this.selectedNode = { layer: l + 1, index: 0, type: 'bias' };
                console.log(`Selected Bias - Layer: ${l + 1}, Index: 0`);
                return;
            }
        }
    }

    drawInfoBox() {
        if (!this.selectedNode || !this.activations) return;
        const { layer, index, type } = this.selectedNode;
        const textLines = this.getInfoBoxText(layer, index, type);
        this.renderInfoBox(textLines);
    }

    /**
     * Generates the text content for the info box.
     * @param {number} layer - The layer index.
     * @param {number} index - The node index.
     * @param {string} type - 'node' or 'bias'.
     * @returns {string[]} An array of strings for the info box.
     */
    getInfoBoxText(layer, index, type) {
        let textLines = [];
        textLines.push(`Layer: ${this.getLayerLabel(layer)}`);
        textLines.push(`Node: ${index}`);

        if (type === 'node') {
            const activationValue = (this.activations[layer] && this.activations[layer].data) ? this.activations[layer].data[index][0] : 0;
            if (layer > 0) {
                const { weightedSum, inputDetails, activationFunctionName } = this.getNodeInfo(layer, index);
                textLines.push(`Inputs: [${inputDetails.map(d => d.input).join(', ')}]`);
                textLines.push(`Weights: [${inputDetails.map(d => d.weight).join(', ')}]`);
                textLines.push(`Bias: ${nf(this.nn.biases[layer - 1].data[index][0], 1, 2)}`);
                textLines.push(`Weighted Sum: ${nf(weightedSum, 1, 2)}`);
                textLines.push(`Activation Func: ${activationFunctionName}`);
                textLines.push(`Final Output: ${nf(activationValue, 1, 2)}`);
            } else {
                textLines.push(`Input Value: ${nf(activationValue, 1, 2)}`);
            }
        } else if (type === 'bias') {
            textLines.push(`Bias Value: ${nf(1.0, 1, 2)}`);
            const nextLayerSize = this.nodePositions[layer].length;
            const biases = this.nn.biases[layer - 1].data;
            for (let i = 0; i < nextLayerSize; i++) {
                textLines.push(`Bias Weight to Node ${i}: ${nf(biases[i][0], 1, 2)}`);
            }
        }
        return textLines;
    }

    /**
     * Gets the label for a given layer.
     * @param {number} layer - The layer index.
     * @returns {string} The layer label.
     */
    getLayerLabel(layer) {
        if (layer === 0) return 'Input';
        if (layer === this.nodePositions.length - 1) return 'Output';
        return `Hidden ${layer}`;
    }

    /**
     * Calculates information for a specific node.
     * @param {number} layer - The layer index.
     * @param {number} index - The node index.
     * @returns {Object} An object containing weighted sum, input details, and activation function name.
     */
    getNodeInfo(layer, index) {
        let weightedSum = 0;
        let inputDetails = [];
        const prevActivations = this.activations[layer - 1].data;
        const weightsToThisNode = this.nn.weights[layer - 1].data[index];

        for (let i = 0; i < weightsToThisNode.length; i++) {
            const input = prevActivations[i][0];
            const weight = weightsToThisNode[i];
            weightedSum += input * weight;
            inputDetails.push({ input: nf(input, 1, 2), weight: nf(weight, 1, 2) });
        }
        weightedSum += this.nn.biases[layer - 1].data[index][0];
        const activationFunctionName = this.nn.activation_functions[layer - 1] ? this.nn.activation_functions[layer - 1].name : 'N/A';
        return { weightedSum, inputDetails, activationFunctionName };
    }

    /**
     * Renders the info box on the screen.
     * @param {string[]} textLines - The text content to display.
     */
    renderInfoBox(textLines) {
        const textPadding = 10;
        const infoX = this.x + 20;
        const infoY = this.y + 20;
        const boxWidth = this.w * 0.25;
        const boxHeight = (textLines.length * 20) + textPadding * 2;

        fill(40, 200);
        stroke(0);
        strokeWeight(1);
        rect(infoX, infoY, boxWidth, boxHeight, 10);

        fill(255);
        textSize(this.r);
        textAlign(LEFT, TOP);
        for (let i = 0; i < textLines.length; i++) {
            text(textLines[i], infoX + textPadding, infoY + textPadding + i * 20);
        }
    }
}