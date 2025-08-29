// nnVisualisation.js - Professional Visualization (Revised)

class NNvisual {
    /**
     * @param {number} x_ - The x-coordinate of the bounding box.
     * @param {number} y_ - The y-coordinate of the bounding box.
     * @param {number} w_ - The width of the bounding box.
     * @param {number} h_ - The height of the bounding box.
     * @param {number} nn_ - The neural network object to visualize.
     */
    constructor(x_, y_, w_, h_, nn_) {
        this.x = x_;
        this.y = y_;
        this.w = w_;
        this.h = h_;
        this.nn = nn_;

        // Ensure hidden_layers is an iterable array before using the spread operator
        const hiddenLayersArray = Array.isArray(this.nn.hidden_layers) ? this.nn.hidden_layers : [this.nn.hidden_layers];

        // Dynamically calculate the node radius based on the largest layer and bounding box
        let layerSizes = [this.nn.input_nodes, ...hiddenLayersArray, this.nn.output_nodes];
        let maxNodesInLayer = Math.max(...layerSizes);
        // Constrain the node radius to a maximum size
        const maxRadius = 30;
        this.r = min(maxRadius, this.h / (maxNodesInLayer * 2 * 2));

        // Constants for cleaner code
        this.NODE_HEIGHT = this.r * 2;
        this.NODE_GAP = this.r * 1.5;

        // State to store the currently selected node
        this.selectedNode = null;
        this.activations = null;

        // Calculate positions once at the start
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
        // Use the same robust method for layerSizes here as in the constructor
        const hiddenLayersArray = Array.isArray(this.nn.hidden_layers) ? this.nn.hidden_layers : [this.nn.hidden_layers];
        let layerSizes = [this.nn.input_nodes, ...hiddenLayersArray, this.nn.output_nodes];
        let numLayers = layerSizes.length;

        // Calculate horizontal spacing to fit within the width
        let layerPadding = this.r * 3; // Add some padding on the sides based on node size
        let layerGap = (this.w - layerPadding * 2) / (max(1, numLayers - 1));

        for (let l = 0; l < numLayers; l++) {
            let numNodes = layerSizes[l];
            let layerX = this.x + layerPadding + l * layerGap;
            let currentLayerPositions = [];

            // Calculate vertical positions for the current layer, including space for the label and bias nodes
            let labelHeight = this.r * 0.6 + 10; // Extra space needed for the label
            let biasHeight = this.r * 1.5 + 20; // Height of the bias node plus some margin
            let totalRequiredHeight = (numNodes * this.NODE_HEIGHT) + ((numNodes - 1) * this.NODE_GAP) + labelHeight + biasHeight;
            let startY = this.y + (this.h - totalRequiredHeight) / 2 + labelHeight;

            for (let i = 0; i < numNodes; i++) {
                currentLayerPositions.push(createVector(layerX, startY + i * (this.NODE_HEIGHT + this.NODE_GAP)));
            }
            nodePositions.push(currentLayerPositions);
        }

        return nodePositions;
    }
    
    /**
     * Calculates the position of each bias node. Bias nodes are placed
     * at the bottom of each layer except the input layer.
     * @returns {Array<p5.Vector>} An array of p5.Vector positions for each bias node.
     */
    calculateBiasPositions() {
        let biasPositions = [];
        const hiddenLayersArray = Array.isArray(this.nn.hidden_layers) ? this.nn.hidden_layers : [this.nn.hidden_layers];
        let layerSizes = [this.nn.input_nodes, ...hiddenLayersArray, this.nn.output_nodes];

        // Bias nodes are for hidden layers and the output layer.
        for (let l = 0; l < layerSizes.length - 1; l++) {
            let numNodesInPreviousLayer = layerSizes[l];
            let prevLayerPositions = this.nodePositions[l];
            
            // The bias node's Y position is that of a hypothetical (n+1)th node in the previous layer.
            let biasY = prevLayerPositions[0].y + numNodesInPreviousLayer * (this.NODE_HEIGHT + this.NODE_GAP);
            
            // The bias node's X position is aligned with the first node of the layer it feeds into.
            let biasX = prevLayerPositions[0].x;

            biasPositions.push(createVector(biasX, biasY));
        }

        return biasPositions;
    }


    /**
     * Renders the neural network visualization.
     * @param {Array<number>} inputs - The input values for the network.
     * @param {Array<number>} outputs - The output values of the network.
     */
    show(inputs, outputs) {
        this.drawBoundingBox();

        // Check if the feedForwardAllLayers method exists on the neural network object
        if (typeof this.nn.feedForwardAllLayers !== 'function') {
            console.error("Error: The provided 'nn' object does not have the 'feedForwardAllLayers' method. Please ensure you are passing a valid NeuralNetwork instance to the NNvisual constructor.");
            return;
        }

        // Run a feedforward pass to get all activation values
        this.activations = this.nn.feedForwardAllLayers(inputs);

        this.drawConnections(this.activations);
        this.drawNodes(this.activations);
        this.drawInfoBox();
    }

    /**
     * Draws a bounding box for the visualization.
     */
    drawBoundingBox() {
        noFill();
        stroke(255);
        strokeWeight(2);
        rect(this.x, this.y, this.w, this.h);
    }

    /**
     * Draws the connections (synapses) between the nodes.
     * @param {Array<object>} activations - The activation values for each layer.
     */
    drawConnections(activations) {
        let numLayers = this.nodePositions.length;
        const CONNECTION_POSITIVE_COLOR = color(150, 250, 150, 150);
        const CONNECTION_NEGATIVE_COLOR = color(250, 150, 150, 150);
        const CONNECTION_NEUTRAL_COLOR = color(200, 200, 200, 80);
        
        // Loop 1: Draw static connection lines and bias lines
        for (let l = 0; l < numLayers - 1; l++) {
            let currentLayerNodes = this.nodePositions[l];
            let nextLayerNodes = this.nodePositions[l + 1];
            let weightsMatrix = this.nn.weights[l];
            
            // Draw connections from bias node
            let biasNodePos = this.biasNodePositions[l];
            let biasConnectionsMatrix = this.nn.biases[l];

            for (let j = 0; j < nextLayerNodes.length; j++) {
                let weight = biasConnectionsMatrix.data[j][0];
                // Clamp the weight to ensure it's within a visual range
                let clampedWeight = min(1, abs(weight));
                let strokeW = map(clampedWeight, 0, 1, 1, 3);
                strokeWeight(strokeW);
                let connectionColor;
                if (weight > 0) {
                    connectionColor = lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_POSITIVE_COLOR, map(weight, 0, 1, 0, 1));
                } else {
                    connectionColor = lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_NEGATIVE_COLOR, map(abs(weight), 0, 1, 0, 1));
                }
                stroke(connectionColor);
                line(biasNodePos.x, biasNodePos.y, nextLayerNodes[j].x, nextLayerNodes[j].y);
            }

            for (let i = 0; i < currentLayerNodes.length; i++) {
                for (let j = 0; j < nextLayerNodes.length; j++) {
                    let weight = weightsMatrix.data[j][i];
                    let startNodePos = currentLayerNodes[i];
                    let endNodePos = nextLayerNodes[j];
                    
                    // Clamp the weight to ensure it's within a visual range
                    let clampedWeight = min(1, abs(weight));
                    let strokeW = map(clampedWeight, 0, 1, 1, 3);
                    strokeWeight(strokeW);
                    let connectionColor;
                    if (weight > 0) {
                        connectionColor = lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_POSITIVE_COLOR, map(weight, 0, 1, 0, 1));
                    } else {
                        connectionColor = lerpColor(CONNECTION_NEUTRAL_COLOR, CONNECTION_NEGATIVE_COLOR, map(abs(weight), 0, 1, 0, 1));
                    }
                    stroke(connectionColor);
                    line(startNodePos.x, startNodePos.y, endNodePos.x, endNodePos.y);
                }
            }
        }
        
        // Loop 2: Draw animated particles on top of the lines
        for (let l = 0; l < numLayers - 1; l++) {
            let currentLayerNodes = this.nodePositions[l];
            let nextLayerNodes = this.nodePositions[l + 1];
            let weightsMatrix = this.nn.weights[l];
            
            // Draw animated particles from bias nodes
            let biasNodePos = this.biasNodePositions[l];
            let biasConnectionsMatrix = this.nn.biases[l];

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
            
            for (let i = 0; i < currentLayerNodes.length; i++) {
                for (let j = 0; j < nextLayerNodes.length; j++) {
                    let weight = weightsMatrix.data[j][i];
                    let startNodePos = currentLayerNodes[i];
                    let endNodePos = nextLayerNodes[j];
                    let activationStrength = (activations[l] && activations[l].data) ? activations[l].data[i][0] : 0;
                    
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

    /**
     * Draws the nodes of the neural network with their activation values.
     * @param {Array<object>} activations - The activation values for each layer.
     */
    drawNodes(activations) {
        noStroke();
        const NEUTRAL_COLOR = color(150, 150, 150);
        const POSITIVE_COLOR = color(100, 200, 100);
        const NEGATIVE_COLOR = color(200, 100, 100);
        const TEXT_COLOR = color(220);

        for (let l = 0; l < this.nodePositions.length; l++) {
            let layerNodes = this.nodePositions[l];
            for (let i = 0; i < layerNodes.length; i++) {
                let pos = layerNodes[i];
                let activationValue = (activations[l] && activations[l].data) ? activations[l].data[i][0] : 0;

                // Determine node color
                let nodeColor;
                if (activationValue > 0) {
                    nodeColor = lerpColor(NEUTRAL_COLOR, POSITIVE_COLOR, map(activationValue, 0, 1, 0, 1));
                } else {
                    nodeColor = lerpColor(NEUTRAL_COLOR, NEGATIVE_COLOR, map(abs(activationValue), 0, 1, 0, 1));
                }

                fill(nodeColor);
                ellipse(pos.x, pos.y, this.r * 2);

                fill(TEXT_COLOR);
                textSize(this.r * 0.5); // Dynamically size text
                textAlign(CENTER, CENTER);
                text(nf(activationValue, 1, 2), pos.x, pos.y);

                if (i === 0) {
                    fill(TEXT_COLOR);
                    textSize(this.r * 0.6); // Dynamically size label text
                    textAlign(CENTER, BOTTOM);

                    let layerLabel;
                    if (l === 0) layerLabel = "Input Layer";
                    else if (l === this.nodePositions.length - 1) layerLabel = "Output Layer";
                    else layerLabel = `Hidden Layer ${l}`;

                    // The label should be drawn a consistent distance above the first node.
                    text(layerLabel, pos.x, this.y + this.r * 0.6); // Adjusted Y position to be relative to the bounding box

                    textSize(this.r * 0.5);
                    textAlign(CENTER, CENTER);
                }
            }
        }
        
        // Draw bias nodes
        for (let l = 0; l < this.biasNodePositions.length; l++) {
            let pos = this.biasNodePositions[l];
            let biasValue = this.nn.biases[l].data[0][0];

            fill(NEUTRAL_COLOR);
            let biasBorderColor;
            if (biasValue > 0) {
                biasBorderColor = POSITIVE_COLOR;
            } else {
                biasBorderColor = NEGATIVE_COLOR;
            }
            stroke(biasBorderColor);
            strokeWeight(3);
            rectMode(CENTER);
            rect(pos.x, pos.y, this.r * 1.5, this.r * 1.5, 5);
            
            fill(TEXT_COLOR);
            noStroke();
            textSize(this.r * 0.5);
            textAlign(CENTER, CENTER);
            text("1.00", pos.x, pos.y);
            
            // Draw bias node label
            fill(TEXT_COLOR);
            textSize(this.r * 0.6);
            textAlign(CENTER, BOTTOM);
            text("Bias", pos.x, pos.y - this.r * 1.5);
        }
        
        // Reset rectMode to CORNER after drawing bias nodes
        rectMode(CORNER);
    }

    /**
     * Handles mouse press events to select a node.
     */
    mousePressed() {
        this.selectedNode = null; // Clear previous selection
        
        // Check for clicks on regular nodes
        for (let l = 0; l < this.nodePositions.length; l++) {
            let layerNodes = this.nodePositions[l];
            for (let i = 0; i < layerNodes.length; i++) {
                let pos = layerNodes[i];
                if (dist(mouseX, mouseY, pos.x, pos.y) < this.r) {
                    this.selectedNode = {
                        layer: l,
                        index: i,
                        type: 'node'
                    };
                    return; // Exit after finding a node
                }
            }
        }
        
        // Check for clicks on bias nodes
        for (let l = 0; l < this.biasNodePositions.length; l++) {
            let pos = this.biasNodePositions[l];
            if (dist(mouseX, mouseY, pos.x, pos.y) < this.r * 1.5) {
                this.selectedNode = {
                    layer: l + 1, // Bias nodes are for hidden layers and output
                    index: 0, // There is only one bias node per layer in this visualization
                    type: 'bias'
                };
                return; // Exit after finding a bias node
            }
        }
    }

    /**
     * Draws an info box for the selected node.
     */
    drawInfoBox() {
        if (!this.selectedNode || !this.activations) {
            return;
        }

        const { layer, index, type } = this.selectedNode;
        const textPadding = 10;
        let infoY, nodePos, activationValue, biasValue = 0, activationFunctionName = '', weightedSum = 0;
        let layerLabel = "";
        let inputDetails = [];
        
        if (type === 'node') {
            nodePos = this.nodePositions[layer][index];
            activationValue = (this.activations[layer] && this.activations[layer].data) ? this.activations[layer].data[index][0] : 0;
            // Correctly map the activation function for the final output layer
            activationFunctionName = (layer === this.nn.weights.length) ? this.nn.activation_functions[layer - 1].name : this.nn.activation_functions[layer].name;
            layerLabel = layer === 0 ? 'Input' : layer === this.nodePositions.length - 1 ? 'Output' : 'Hidden ' + layer;
            
            if (layer > 0) {
                let prevActivations = this.activations[layer - 1].data;
                let weightsToThisNode = this.nn.weights[layer - 1].data[index];
                
                for (let i = 0; i < weightsToThisNode.length; i++) {
                    let input = prevActivations[i][0];
                    let weight = weightsToThisNode[i];
                    weightedSum += input * weight;
                    inputDetails.push({input: nf(input, 1, 2), weight: nf(weight, 1, 2)});
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
            weightedSum = 1.0; // The activation itself
        }
        
        let textLines = [];
        textLines.push(`Layer: ${layerLabel}`);
        if (type === 'node') textLines.push(`Node: ${index}`);
        else if (type === 'bias') textLines.push(`Bias Value: ${nf(activationValue, 1, 2)}`);
        
        if (type === 'node' && layer > 0) {
            let inputsText = `Inputs: [${inputDetails.map(d => d.input).join(', ')}]`;
            let weightsText = `Weights: [${inputDetails.map(d => d.weight).join(', ')}]`;
            textLines.push(inputsText);
            textLines.push(weightsText);
            textLines.push(`Bias: ${nf(biasValue, 1, 2)}`);
            textLines.push(`Weighted Sum: ${nf(weightedSum, 1, 2)}`);
            textLines.push(`Final Output: ${nf(activationValue, 1, 2)}`);
            textLines.push(`Activation Func: ${activationFunctionName}`);
        } else if (type === 'node' && layer === 0) {
            textLines.push(`Input Value: ${nf(activationValue, 1, 2)}`);
        } else if (type === 'bias') {
            textLines.push(`Bias Weight: ${nf(biasValue, 1, 2)}`);
        }

        // Calculate dynamic box width based on content
        let boxWidth = 0;
        for (let line of textLines) {
             let currentWidth = textWidth(line);
             if (currentWidth > boxWidth) {
                 boxWidth = currentWidth;
             }
        }
        boxWidth += textPadding * 2; // Add padding to the final width
        let boxHeight = (textLines.length * 20) + textPadding * 2;

        // Position the info box relative to the node
        let infoX = nodePos.x + this.r + 20;
        infoY = nodePos.y - boxHeight / 2;
        
        // Adjust position if it would go off the right side
        if (infoX + boxWidth > this.x + this.w) {
            infoX = nodePos.x - this.r - 20 - boxWidth;
        }
        
        // Draw the info box background
        fill(40);
        stroke(200);
        strokeWeight(1); 
        rect(infoX, infoY, boxWidth, boxHeight, 10);

        // Draw the text
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