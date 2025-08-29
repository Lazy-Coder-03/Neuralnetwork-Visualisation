// nnVisualisation.js - Professional Visualization (Revised)

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
            this.w = w_;
            this.h = h_;
        } else {
            this.x = x_;
            this.y = y_;
            this.w = w_;
            this.h = h_;
        }
        this.nn = nn_;
        this.showinfobox = options.showinfobox !== undefined ? options.showinfobox : true;

        // Ensure hidden_layers is iterable
        const hiddenLayersArray = Array.isArray(this.nn.hidden_layers) ? this.nn.hidden_layers : [this.nn.hidden_layers];
        let layerSizes = [this.nn.input_nodes, ...hiddenLayersArray, this.nn.output_nodes];
        let maxNodesInLayer = Math.max(...layerSizes);

        // Dynamically calculate node radius
        const maxRadius = 30;
        this.r = min(maxRadius, this.h / (maxNodesInLayer * 2 * 2));

        this.NODE_HEIGHT = this.r * 2;
        this.NODE_GAP = this.r * 1.5;

        this.selectedNode = null;
        this.activations = null;

        this.nodePositions = this.calculateLayerPositions();
        this.biasNodePositions = this.calculateBiasPositions();
    }

    /**
     * Calculates the position of each node in the neural network based on the bounding box.
     * This ensures the network fits properly within the specified area.
     * @returns {Array<Array<p5.Vector>>} An array of arrays containing the p5.Vector positions for each node.
     */
    calculateLayerPositions() {
        let nodePositions = [];
        const hiddenLayersArray = Array.isArray(this.nn.hidden_layers) ? this.nn.hidden_layers : [this.nn.hidden_layers];
        let layerSizes = [this.nn.input_nodes, ...hiddenLayersArray, this.nn.output_nodes];
        let numLayers = layerSizes.length;

        // Calculate horizontal positions to ensure consistent spacing
        let layerXs = [];
        let layerPadding = this.r * 3;
        let totalHorizontalSpace = this.w - layerPadding * 2;
        let numGaps = numLayers - 1;

        for (let l = 0; l < numLayers; l++) {
            let numNodes = layerSizes[l];
            let layerX;

            const GRID_THRESHOLD = 20;
            if (numNodes > GRID_THRESHOLD) {
                let numCols = ceil(sqrt(numNodes));
                let totalGridWidth = (numCols * this.NODE_HEIGHT) + ((numCols - 1) * this.NODE_GAP);
                layerX = this.x + layerPadding + (l / max(1, numGaps)) * (totalHorizontalSpace);
            } else {
                layerX = this.x + layerPadding + (l / max(1, numGaps)) * totalHorizontalSpace;
            }
            layerXs.push(layerX);
        }

        for (let l = 0; l < numLayers; l++) {
            let numNodes = layerSizes[l];
            let layerX = layerXs[l];
            let currentLayerPositions = [];

            let labelHeight = this.r * 0.6 + 10;
            let biasHeight = this.r * 1.5 + 20;

            const GRID_THRESHOLD = 20;
            if (numNodes > GRID_THRESHOLD) {
                let numCols = ceil(sqrt(numNodes));
                let numRows = ceil(numNodes / numCols);

                let totalGridWidth = (numCols * this.NODE_HEIGHT) + ((numCols - 1) * this.NODE_GAP);
                let totalGridHeight = (numRows * this.NODE_HEIGHT) + ((numRows - 1) * this.NODE_GAP);
                let totalRequiredHeight = totalGridHeight + labelHeight + biasHeight;

                let startY = this.y + (this.h - totalRequiredHeight) / 2 + labelHeight;
                let gridStartX = layerX - totalGridWidth / 2;

                for (let i = 0; i < numNodes; i++) {
                    let row = floor(i / numCols);
                    let col = i % numCols;
                    let nodeX = gridStartX + col * (this.NODE_HEIGHT + this.NODE_GAP);
                    let nodeY = startY + row * (this.NODE_HEIGHT + this.NODE_GAP);
                    currentLayerPositions.push(createVector(nodeX, nodeY));
                }
            } else {
                let totalRequiredHeight = (numNodes * this.NODE_HEIGHT) + ((numNodes - 1) * this.NODE_GAP) + labelHeight + biasHeight;
                let startY = this.y + (this.h - totalRequiredHeight) / 2 + labelHeight;

                for (let i = 0; i < numNodes; i++) {
                    currentLayerPositions.push(createVector(layerX, startY + i * (this.NODE_HEIGHT + this.NODE_GAP)));
                }
            }

            nodePositions.push(currentLayerPositions);
        }

        return nodePositions;
    }

    calculateBiasPositions() {
        let biasPositions = [];
        const hiddenLayersArray = Array.isArray(this.nn.hidden_layers) ? this.nn.hidden_layers : [this.nn.hidden_layers];
        let layerSizes = [this.nn.input_nodes, ...hiddenLayersArray, this.nn.output_nodes];

        // Find the max Y position of all nodes across all layers
        let maxYOfAllNodes = 0;
        for (let l = 0; l < this.nodePositions.length; l++) {
            for (let i = 0; i < this.nodePositions[l].length; i++) {
                if (this.nodePositions[l][i].y > maxYOfAllNodes) {
                    maxYOfAllNodes = this.nodePositions[l][i].y;
                }
            }
        }

        // Use this single maxY value for all bias nodes
        let commonBiasY = maxYOfAllNodes + this.NODE_HEIGHT + this.NODE_GAP;

        for (let l = 0; l < layerSizes.length - 1; l++) {
            let prevLayerPositions = this.nodePositions[l];

            // The biasX position is aligned with the first node of the layer it feeds into.
            let biasX = prevLayerPositions[0].x;

            biasPositions.push(createVector(biasX, commonBiasY));
        }

        return biasPositions;
    }

    show(inputs) {
        this.drawBoundingBox();

        if (typeof this.nn.feedForwardAllLayers !== 'function') {
            console.error("Error: The provided 'nn' object does not have the 'feedForwardAllLayers' method.");
            return;
        }

        this.activations = this.nn.feedForwardAllLayers(inputs);

        this.drawConnections();
        this.drawNodes();
        if (this.showinfobox) {
            this.drawInfoBox();
        }
    }

    drawBoundingBox() {
        noFill();
        stroke(255);
        strokeWeight(2);
        rect(this.x, this.y, this.w, this.h);
    }

    drawConnections() {
        let numLayers = this.nodePositions.length;
        const CONNECTION_POSITIVE_COLOR = color(150, 250, 150, 150);
        const CONNECTION_NEGATIVE_COLOR = color(250, 150, 150, 150);
        const CONNECTION_NEUTRAL_COLOR = color(50, 50, 50, 80);
        const WEIGHT_THRESHOLD = 0.5;

        if (this.selectedNode && this.selectedNode.type === 'node' && this.selectedNode.layer === numLayers - 1) {
            let outputNodeIndex = this.selectedNode.index;
            let selectedNodesToDraw = new Set();
            selectedNodesToDraw.add(`${numLayers - 1},${outputNodeIndex}`);

            // Recursively find all connected nodes in previous layers
            for (let l = numLayers - 1; l > 0; l--) {
                let prevLayerNodes = this.nodePositions[l - 1];
                let nextLayerNodes = this.nodePositions[l];
                let weightsMatrix = this.nn.weights[l - 1];
                let biasConnectionsMatrix = this.nn.biases[l - 1];
                let newNodesInPrevLayer = new Set();

                for (let j = 0; j < nextLayerNodes.length; j++) {
                    if (selectedNodesToDraw.has(`${l},${j}`)) {
                        // Check bias connection to this node
                        let biasWeight = biasConnectionsMatrix.data[j][0];
                        if (abs(biasWeight) > WEIGHT_THRESHOLD) {
                            let biasNodePos = this.biasNodePositions[l - 1];
                            let endPos = nextLayerNodes[j];

                            // Draw bias connection
                            let connectionColor = biasWeight > 0 ? lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_POSITIVE_COLOR, map(biasWeight, 0, 1, 0, 1)) : lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_NEGATIVE_COLOR, map(abs(biasWeight), 0, 1, 0, 1));
                            let constrainedWeight = constrain(biasWeight, -1, 1);
                            let strokeW = map(abs(constrainedWeight), 0, 1, 1, 3);
                            strokeWeight(strokeW);
                            stroke(connectionColor);
                            line(biasNodePos.x, biasNodePos.y, endPos.x, endPos.y);

                            // Draw particle
                            let animationProgress = (frameCount * 0.02 + abs(biasWeight) * 5) % 1;
                            let animX = lerp(biasNodePos.x, endPos.x, animationProgress);
                            let animY = lerp(biasNodePos.y, endPos.y, animationProgress);
                            noStroke();
                            let particleColor = lerpColor(color(0, 0, 0, 0), color(255, 255, 255, 200), abs(biasWeight));
                            fill(particleColor);
                            ellipse(animX, animY, strokeW + 2, strokeW + 2);
                        }

                        // Check connections from previous nodes to this node
                        for (let i = 0; i < prevLayerNodes.length; i++) {
                            let weight = weightsMatrix.data[j][i];
                            if (abs(weight) > WEIGHT_THRESHOLD) {
                                newNodesInPrevLayer.add(`${l - 1},${i}`);
                                let startNodePos = prevLayerNodes[i];
                                let endNodePos = nextLayerNodes[j];

                                // Draw connection
                                let clampedWeight = min(0, abs(weight));
                                let strokeW = map(clampedWeight, 0, 1, 1, 3);
                                let connectionColor = weight > 0 ? lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_POSITIVE_COLOR, map(weight, 0, 1, 0, 1)) : lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_NEGATIVE_COLOR, map(abs(weight), 0, 1, 0, 1));
                                strokeWeight(strokeW);
                                stroke(connectionColor);
                                line(startNodePos.x, startNodePos.y, endNodePos.x, endNodePos.y);

                                // Draw particle
                                let activationStrength = (this.activations[l - 1] && this.activations[l - 1].data) ? this.activations[l - 1].data[i][0] : 0;
                                let animationProgress = (frameCount * 0.02 + activationStrength * 5) % 1;
                                let animX = lerp(startNodePos.x, endNodePos.x, animationProgress);
                                let animY = lerp(startNodePos.y, endNodePos.y, animationProgress);
                                noStroke();
                                let particleColor = lerpColor(color(0, 0, 0, 0), color(255, 255, 255, 200), abs(activationStrength));
                                fill(particleColor);
                                ellipse(animX, animY, strokeW + 2, strokeW + 2);
                            }
                        }
                    }
                }
                selectedNodesToDraw = newNodesInPrevLayer;
            }

        } else {
            // Original code to draw all connections if no output node is selected
            // This includes all connections, both strong and weak, and their particles.
            for (let l = 0; l < numLayers - 1; l++) {
                let currentLayerNodes = this.nodePositions[l];
                let nextLayerNodes = this.nodePositions[l + 1];
                let weightsMatrix = this.nn.weights[l];
                let biasNodePos = this.biasNodePositions[l];
                let biasConnectionsMatrix = this.nn.biases[l];

                for (let j = 0; j < nextLayerNodes.length; j++) {
                    let weight = biasConnectionsMatrix.data[j][0];
                    let clampedWeight = min(0, abs(weight));
                    let strokeW = map(clampedWeight, 0, 1, 1, 3);
                    let connectionColor = weight > 0
                        ? lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_POSITIVE_COLOR, map(weight, 0, 1, 0, 1))
                        : lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_NEGATIVE_COLOR, map(abs(weight), 0, 1, 0, 1));

                    strokeWeight(strokeW);
                    stroke(connectionColor);
                    line(biasNodePos.x, biasNodePos.y, nextLayerNodes[j].x, nextLayerNodes[j].y);

                    // Particle for bias connection
                    let animationProgress = (frameCount * 0.02 + abs(weight) * 5) % 1;
                    let animX = lerp(biasNodePos.x, nextLayerNodes[j].x, animationProgress);
                    let animY = lerp(biasNodePos.y, nextLayerNodes[j].y, animationProgress);
                    noStroke();
                    let particleColor = lerpColor(color(0, 0, 0, 0), color(255, 255, 255, 200), abs(weight));
                    fill(particleColor);
                    ellipse(animX, animY, strokeW + 2, strokeW + 2);
                }

                for (let i = 0; i < currentLayerNodes.length; i++) {
                    for (let j = 0; j < nextLayerNodes.length; j++) {
                        let weight = weightsMatrix.data[j][i];
                        let startNodePos = currentLayerNodes[i];
                        let endNodePos = nextLayerNodes[j];
                        let clampedWeight = min(0, abs(weight));
                        let strokeW = map(clampedWeight, 0, 1, 1, 3);
                        let connectionColor = weight > 0
                            ? lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_POSITIVE_COLOR, map(weight, 0, 1, 0, 1))
                            : lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_NEGATIVE_COLOR, map(abs(weight), 0, 1, 0, 1));

                        strokeWeight(strokeW);
                        stroke(connectionColor);
                        line(startNodePos.x, startNodePos.y, endNodePos.x, endNodePos.y);

                        // Particle for regular connection
                        let activationStrength = (this.activations[l] && this.activations[l].data) ? this.activations[l].data[i][0] : 0;
                        let animationProgress = (frameCount * 0.02 + activationStrength * 5) % 1;
                        let animX = lerp(startNodePos.x, endNodePos.x, animationProgress);
                        let animY = lerp(startNodePos.y, endNodePos.y, animationProgress);
                        noStroke();
                        let particleColor = lerpColor(color(0, 0, 0, 0), color(255, 255, 255, 200), abs(activationStrength));
                        fill(particleColor);
                        ellipse(animX, animY, strokeW + 2, strokeW + 2);
                    }
                }
            }
        }
    }

    drawNodes() {
        const NEUTRAL_COLOR = color(150, 150, 150);
        const HEATMAP_POSITIVE_COLOR = color(255, 100, 100);
        const HEATMAP_NEGATIVE_COLOR = color(100, 100, 255);
        const TEXT_COLOR = color(220);

        for (let l = 0; l < this.nodePositions.length; l++) {
            let layerNodes = this.nodePositions[l];
            for (let i = 0; i < layerNodes.length; i++) {
                let pos = layerNodes[i];
                let activationValue = (this.activations[l] && this.activations[l].data) ? this.activations[l].data[i][0] : 0;
                let nodeColor;

                if (activationValue > 0) {
                    nodeColor = lerpColor(NEUTRAL_COLOR, HEATMAP_POSITIVE_COLOR, map(activationValue, 0, 1, 0, 1));
                } else {
                    nodeColor = lerpColor(NEUTRAL_COLOR, HEATMAP_NEGATIVE_COLOR, map(abs(activationValue), 0, 1, 0, 1));
                }

                noStroke();
                fill(nodeColor);
                ellipse(pos.x, pos.y, this.r * 2);

                // Highlight selected node
                if (this.selectedNode && this.selectedNode.type === 'node' && this.selectedNode.layer === l && this.selectedNode.index === i) {
                    noFill();
                    stroke(255, 255, 0); // Yellow highlight
                    strokeWeight(3);
                    ellipse(pos.x, pos.y, this.r * 2 + 6);
                }

                fill(TEXT_COLOR);
                textSize(this.r * 0.75);
                stroke(0);
                strokeWeight(this.r * 0.2);
                textAlign(CENTER, CENTER);
                text(nf(activationValue, 1, 2), pos.x, pos.y);
            }
        }

        for (let l = 0; l < this.biasNodePositions.length; l++) {
            let pos = this.biasNodePositions[l];
            let biasValue = this.nn.biases[l].data[0][0];
            fill(NEUTRAL_COLOR);
            stroke(biasValue > 0 ? color(100, 200, 100) : color(200, 100, 100));
            strokeWeight(3);
            rectMode(CENTER);
            rect(pos.x, pos.y, this.r * 1.5, this.r * 1.5, 5);

            // Highlight selected bias node
            if (this.selectedNode && this.selectedNode.type === 'bias' && this.selectedNode.layer === l + 1) {
                noFill();
                stroke(255, 255, 0); // Yellow highlight
                strokeWeight(3);
                rect(pos.x, pos.y, this.r * 1.5 + 6, this.r * 1.5 + 6, 5);
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

    mousePressed() {
        this.selectedNode = null;
        for (let l = 0; l < this.nodePositions.length; l++) {
            let layerNodes = this.nodePositions[l];
            for (let i = 0; i < layerNodes.length; i++) {
                let pos = layerNodes[i];
                if (dist(mouseX, mouseY, pos.x, pos.y) < this.r) {
                    this.selectedNode = { layer: l, index: i, type: 'node' };
                    console.log(`Selected Node - Layer: ${l}, Index: ${i}`);
                    return;
                }
            }
        }
        for (let l = 0; l < this.biasNodePositions.length; l++) {
            let pos = this.biasNodePositions[l];
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
        const textPadding = 10;
        const infoX = this.x + 20; // Static to top-left
        const infoY = this.y + 20; // Static to top-left
        let activationValue, biasValue = 0, activationFunctionName = '', weightedSum = 0;
        let layerLabel = "";
        let inputDetails = [];

        let textLines = [];
        textLines.push(`Layer: ${layer === 0 ? 'Input' : layer === this.nodePositions.length - 1 ? 'Output' : `Hidden ${layer}`}`);
        textLines.push(`Node: ${index}`);

        if (type === 'node') {
            activationValue = (this.activations[layer] && this.activations[layer].data) ? this.activations[layer].data[index][0] : 0;
            activationFunctionName = layer > 0 ? (this.nn.activation_functions[layer - 1] ? this.nn.activation_functions[layer - 1].name : 'N/A') : 'N/A (Input)';
            layerLabel = layer === 0 ? 'Input' : layer === this.nodePositions.length - 1 ? 'Output' : `Hidden ${layer}`;

            if (layer > 0) {
                let prevActivations = this.activations[layer - 1].data;
                let weightsToThisNode = this.nn.weights[layer - 1].data[index];

                for (let i = 0; i < weightsToThisNode.length; i++) {
                    let input = prevActivations[i][0];
                    let weight = weightsToThisNode[i];
                    weightedSum += input * weight;
                    inputDetails.push({ input: nf(input, 1, 2), weight: nf(weight, 1, 2) });
                }
                biasValue = this.nn.biases[layer - 1].data[index][0];
                weightedSum += biasValue;

                textLines.push(`Inputs: [${inputDetails.map(d => d.input).join(', ')}]`);
                textLines.push(`Weights: [${inputDetails.map(d => d.weight).join(', ')}]`);
                textLines.push(`Bias: ${nf(biasValue, 1, 2)}`);
                textLines.push(`Weighted Sum: ${nf(weightedSum, 1, 2)}`);
                textLines.push(`Activation Func: ${activationFunctionName}`);
                textLines.push(`Final Output: ${nf(activationValue, 1, 2)}`);

            } else {
                textLines.push(`Input Value: ${nf(activationValue, 1, 2)}`);
            }
        } else if (type === 'bias') {
            textLines.push(`Bias Value: ${nf(1.0, 1, 2)}`);
            let nextLayerSize = this.nodePositions[layer].length;
            let biases = this.nn.biases[layer - 1].data;

            for (let i = 0; i < nextLayerSize; i++) {
                let biasWeight = biases[i][0];
                textLines.push(`Bias Weight to Node ${i}: ${nf(biasWeight, 1, 2)}`);
            }
        }

        let boxWidth = this.w * 0.25;
        let boxHeight = (textLines.length * 20) + textPadding * 2;

        fill(40, 200);
        stroke(0);
        strokeWeight(1);
        rect(infoX, infoY, boxWidth, boxHeight, 10);

        fill(255);
        textSize(this.r);
        textAlign(LEFT, TOP);
        let textX = infoX + textPadding;
        let textY = infoY + textPadding;

        for (let i = 0; i < textLines.length; i++) {
            text(textLines[i], textX, textY + i * 20);
        }
    }
}