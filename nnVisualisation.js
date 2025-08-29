// nnVisualisation.js - Professional Visualization (Revised)

class NNvisual {
    /**
     * @param {number} x_ - The x-coordinate of the bounding box.
     * @param {number} y_ - The y-coordinate of the bounding box.
     * @param {number} w_ - The width of the bounding box.
     * @param {number} h_ - The height of the bounding box.
     * @param {NeuralNetwork} nn_ - The neural network object to visualize.
     */
    constructor(x_, y_, w_, h_, nn_) {
        this.x = x_;
        this.y = y_;
        this.w = w_;
        this.h = h_;
        this.nn = nn_;

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

    calculateLayerPositions() {
        let nodePositions = [];
        const hiddenLayersArray = Array.isArray(this.nn.hidden_layers) ? this.nn.hidden_layers : [this.nn.hidden_layers];
        let layerSizes = [this.nn.input_nodes, ...hiddenLayersArray, this.nn.output_nodes];
        let numLayers = layerSizes.length;

        let layerPadding = this.r * 3;
        let layerGap = (this.w - layerPadding * 2) / (max(1, numLayers - 1));

        for (let l = 0; l < numLayers; l++) {
            let numNodes = layerSizes[l];
            let layerX = this.x + layerPadding + l * layerGap;
            let currentLayerPositions = [];

            let labelHeight = this.r * 0.6 + 10;
            let biasHeight = this.r * 1.5 + 20;
            let totalRequiredHeight = (numNodes * this.NODE_HEIGHT) + ((numNodes - 1) * this.NODE_GAP) + labelHeight + biasHeight;
            let startY = this.y + (this.h - totalRequiredHeight) / 2 + labelHeight;

            for (let i = 0; i < numNodes; i++) {
                currentLayerPositions.push(createVector(layerX, startY + i * (this.NODE_HEIGHT + this.NODE_GAP)));
            }
            nodePositions.push(currentLayerPositions);
        }

        return nodePositions;
    }

    calculateBiasPositions() {
        let biasPositions = [];
        const hiddenLayersArray = Array.isArray(this.nn.hidden_layers) ? this.nn.hidden_layers : [this.nn.hidden_layers];
        let layerSizes = [this.nn.input_nodes, ...hiddenLayersArray, this.nn.output_nodes];

        for (let l = 0; l < layerSizes.length - 1; l++) {
            let numNodesInPreviousLayer = layerSizes[l];
            let prevLayerPositions = this.nodePositions[l];
            let biasY = prevLayerPositions[0].y + numNodesInPreviousLayer * (this.NODE_HEIGHT + this.NODE_GAP);
            let biasX = prevLayerPositions[0].x;
            biasPositions.push(createVector(biasX, biasY));
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
        this.drawInfoBox();
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
        const CONNECTION_NEUTRAL_COLOR = color(200, 200, 200, 80);

        for (let l = 0; l < numLayers - 1; l++) {
            let currentLayerNodes = this.nodePositions[l];
            let nextLayerNodes = this.nodePositions[l + 1];
            let weightsMatrix = this.nn.weights[l];
            let biasNodePos = this.biasNodePositions[l];
            let biasConnectionsMatrix = this.nn.biases[l];

            // Bias connections
            for (let j = 0; j < nextLayerNodes.length; j++) {
                let weight = biasConnectionsMatrix.data[j][0];
                let clampedWeight = min(1, abs(weight));
                let strokeW = map(clampedWeight, 0, 1, 1, 3);
                strokeWeight(strokeW);
                let connectionColor = weight > 0
                    ? lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_POSITIVE_COLOR, map(weight, 0, 1, 0, 1))
                    : lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_NEGATIVE_COLOR, map(abs(weight), 0, 1, 0, 1));
                stroke(connectionColor);
                line(biasNodePos.x, biasNodePos.y, nextLayerNodes[j].x, nextLayerNodes[j].y);
            }

            // Node connections
            for (let i = 0; i < currentLayerNodes.length; i++) {
                for (let j = 0; j < nextLayerNodes.length; j++) {
                    let weight = weightsMatrix.data[j][i];
                    let startNodePos = currentLayerNodes[i];
                    let endNodePos = nextLayerNodes[j];
                    let clampedWeight = min(1, abs(weight));
                    let strokeW = map(clampedWeight, 0, 1, 1, 3);
                    strokeWeight(strokeW);
                    let connectionColor = weight > 0
                        ? lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_POSITIVE_COLOR, map(weight, 0, 1, 0, 1))
                        : lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_NEGATIVE_COLOR, map(abs(weight), 0, 1, 0, 1));
                    stroke(connectionColor);
                    line(startNodePos.x, startNodePos.y, endNodePos.x, endNodePos.y);
                }
            }
        }

        // Animated particles
        for (let l = 0; l < numLayers - 1; l++) {
            let currentLayerNodes = this.nodePositions[l];
            let nextLayerNodes = this.nodePositions[l + 1];
            let weightsMatrix = this.nn.weights[l];
            let biasNodePos = this.biasNodePositions[l];
            let biasConnectionsMatrix = this.nn.biases[l];

            // Bias particles
            for (let j = 0; j < nextLayerNodes.length; j++) {
                let weight = biasConnectionsMatrix.data[j][0];
                let strokeW = map(min(1, abs(weight)), 0, 1, 1, 3);
                let startPos = biasNodePos;
                let endPos = nextLayerNodes[j];
                let animationProgress = (frameCount * 0.05 + abs(weight) * 2) % 1;
                let animX = lerp(startPos.x, endPos.x, animationProgress);
                let animY = lerp(startPos.y, endPos.y, animationProgress);
                noStroke();
                let particleColor = lerpColor(color(0, 0, 0, 0), color(255, 255, 255, 200), abs(weight));
                fill(particleColor);
                ellipse(animX, animY, strokeW + 2, strokeW + 2);
            }

            // Node particles
            for (let i = 0; i < currentLayerNodes.length; i++) {
                for (let j = 0; j < nextLayerNodes.length; j++) {
                    let weight = weightsMatrix.data[j][i];
                    let startNodePos = currentLayerNodes[i];
                    let endNodePos = nextLayerNodes[j];
                    let activationStrength = (this.activations[l] && this.activations[l].data) ? this.activations[l].data[i][0] : 0;
                    let animationProgress = (frameCount * 0.05 + activationStrength * 2) % 1;
                    let animX = lerp(startNodePos.x, endNodePos.x, animationProgress);
                    let animY = lerp(startNodePos.y, endNodePos.y, animationProgress);
                    noStroke();
                    let particleColor = lerpColor(color(0, 0, 0, 0), color(255, 255, 255, 200), abs(activationStrength));
                    fill(particleColor);
                    let strokeW = map(min(1, abs(weight)), 0, 1, 1, 3);
                    ellipse(animX, animY, strokeW + 2, strokeW + 2);
                }
            }
        }
    }

    drawNodes() {
        noStroke();
        const NEUTRAL_COLOR = color(150, 150, 150);
        const POSITIVE_COLOR = color(100, 200, 100);
        const NEGATIVE_COLOR = color(200, 100, 100);
        const TEXT_COLOR = color(220);

        for (let l = 0; l < this.nodePositions.length; l++) {
            let layerNodes = this.nodePositions[l];
            for (let i = 0; i < layerNodes.length; i++) {
                let pos = layerNodes[i];
                let activationValue = (this.activations[l] && this.activations[l].data) ? this.activations[l].data[i][0] : 0;
                let nodeColor = activationValue > 0
                    ? lerpColor(NEUTRAL_COLOR, POSITIVE_COLOR, map(activationValue, 0, 1, 0, 1))
                    : lerpColor(NEUTRAL_COLOR, NEGATIVE_COLOR, map(abs(activationValue), 0, 1, 0, 1));
                fill(nodeColor);
                ellipse(pos.x, pos.y, this.r * 2);
                fill(TEXT_COLOR);
                textSize(this.r * 0.5);
                textAlign(CENTER, CENTER);
                text(nf(activationValue, 1, 2), pos.x, pos.y);

                if (i === 0) {
                    fill(TEXT_COLOR);
                    textSize(this.r * 0.6);
                    textAlign(CENTER, BOTTOM);
                    let layerLabel = l === 0 ? "Input Layer" : l === this.nodePositions.length - 1 ? "Output Layer" : `Hidden Layer ${l}`;
                    text(layerLabel, pos.x, this.y + this.r * 0.6);
                    textSize(this.r * 0.5);
                    textAlign(CENTER, CENTER);
                }
            }
        }

        // Bias nodes
        for (let l = 0; l < this.biasNodePositions.length; l++) {
            let pos = this.biasNodePositions[l];
            let biasValue = this.nn.biases[l].data[0][0];
            fill(NEUTRAL_COLOR);
            stroke(biasValue > 0 ? POSITIVE_COLOR : NEGATIVE_COLOR);
            strokeWeight(3);
            rectMode(CENTER);
            rect(pos.x, pos.y, this.r * 1.5, this.r * 1.5, 5);
            fill(TEXT_COLOR);
            noStroke();
            textSize(this.r * 0.5);
            textAlign(CENTER, CENTER);
            text("1.00", pos.x, pos.y);
            fill(TEXT_COLOR);
            textSize(this.r * 0.6);
            textAlign(CENTER, BOTTOM);
            text("Bias", pos.x, pos.y - this.r * 1.5);
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
                    return;
                }
            }
        }
        for (let l = 0; l < this.biasNodePositions.length; l++) {
            let pos = this.biasNodePositions[l];
            if (dist(mouseX, mouseY, pos.x, pos.y) < this.r * 1.5) {
                this.selectedNode = { layer: l + 1, index: 0, type: 'bias' };
                return;
            }
        }
    }

    drawInfoBox() {
        if (!this.selectedNode || !this.activations) return;
        const { layer, index, type } = this.selectedNode;
        const textPadding = 10;
        let nodePos, activationValue, biasValue = 0, activationFunctionName = '', weightedSum = 0;
        let layerLabel = "";
        let inputDetails = [];

        if (type === 'node') {
            nodePos = this.nodePositions[layer][index];
            activationValue = (this.activations[layer] && this.activations[layer].data) ? this.activations[layer].data[index][0] : 0;
            activationFunctionName = layer > 0
                ? this.nn.activation_functions[layer - 1] ? this.nn.activation_functions[layer - 1].name : 'N/A'
                : 'N/A (Input)';
            layerLabel = layer === 0 ? 'Input' : layer === this.nodePositions.length - 1 ? 'Output' : 'Hidden ' + layer;

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
            } else {
                weightedSum = activationValue;
            }
        } else if (type === 'bias') {
            nodePos = this.biasNodePositions[layer - 1];
            activationValue = 1.0;
            biasValue = this.nn.biases[layer - 1].data[0][0];
            activationFunctionName = 'N/A (Bias)';
            layerLabel = layer === this.biasNodePositions.length ? 'Output Layer Bias' : 'Hidden Layer Bias ' + layer;
            weightedSum = 1.0;
        }

        let textLines = [];
        textLines.push(`Layer: ${layerLabel}`);
        if (type === 'node') textLines.push(`Node: ${index}`);
        else if (type === 'bias') textLines.push(`Bias Value: ${nf(activationValue, 1, 2)}`);

        if (type === 'node' && layer > 0) {
            textLines.push(`Inputs: [${inputDetails.map(d => d.input).join(', ')}]`);
            textLines.push(`Weights: [${inputDetails.map(d => d.weight).join(', ')}]`);
            textLines.push(`Bias: ${nf(biasValue, 1, 2)}`);
            textLines.push(`Weighted Sum: ${nf(weightedSum, 1, 2)}`);
            textLines.push(`Final Output: ${nf(activationValue, 1, 2)}`);
            textLines.push(`Activation Func: ${activationFunctionName}`);
        } else if (type === 'node' && layer === 0) {
            textLines.push(`Input Value: ${nf(activationValue, 1, 2)}`);
        } else if (type === 'bias') {
            textLines.push(`Bias Weight: ${nf(biasValue, 1, 2)}`);
        }

        let boxWidth = 0;
        for (let line of textLines) {
            let currentWidth = textWidth(line);
            if (currentWidth > boxWidth) boxWidth = currentWidth;
        }
        boxWidth += textPadding * 2;
        let boxHeight = (textLines.length * 20) + textPadding * 2;
        let infoX = nodePos.x + this.r + 20;
        let infoY = nodePos.y - boxHeight / 2;
        if (infoX + boxWidth > this.x + this.w) infoX = nodePos.x - this.r - 20 - boxWidth;

        fill(40);
        stroke(200);
        strokeWeight(1);
        rect(infoX, infoY, boxWidth, boxHeight, 10);

        fill(255);
        textSize(14);
        textAlign(LEFT, TOP);
        let textX = infoX + textPadding;
        let textY = infoY + textPadding;
        for (let i = 0; i < textLines.length; i++) {
            text(textLines[i], textX, textY + i * 20);
        }
    }
}
