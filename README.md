# Neural Network Visualization

A modern, interactive web application for visualizing and training simple neural networks on classic logic and encoding tasks. Built with JavaScript and p5.js, this project provides a live, animated view of neural network learning, with customizable architecture and datasets.

## Features

- **Live Neural Network Visualization:** See activations, weights, and node details in real time as the network trains.
- **Customizable Architecture:** Add/remove hidden layers and nodes (up to 20 per layer) via the UI.
- **Multiple Datasets:** Includes classic tasks like Encoder/Decoder, AND/XOR gates, Adder, and Subtractor.
- **Training Controls:** Adjust epochs per frame, pause/resume training, and select test cases.
- **Node Inspection:** Click any node to view its details and activation path.
- **Responsive UI:** Built with Tailwind CSS for a clean, modern look.

## Demo

To run locally:

1. **Install Python (3.x recommended).**
2. **Start the local server:**
   ```powershell
   python setup.py
   ```
   Or specify a port:
   ```powershell
   python setup.py 8080
   ```
3. **Open your browser:**
   Visit [http://localhost:8000](http://localhost:8000) (or your chosen port).

## File Structure

- `index.html` — Main HTML page and UI layout
- `style.css` — Custom styles (in addition to Tailwind)
- `matrix.js` — Matrix math utilities
- `neuralNetwork.js` — Core neural network implementation
- `nnVisualisation.js` — Visualization logic (p5.js)
- `sketch.js` — Main app logic and UI event handling
- `trainingData.json` — Predefined datasets for training/testing
- `setup.py` — Simple Python HTTP server for local development

## Datasets

- **Encoder 3bit**
- **Decoder 3bit**
- **AND Gate**
- **XOR Gate**
- **2bit Adder + carry bit**
- **2 bit Subtractor**

Each dataset includes network configuration and sample data for training/testing.

## Customization

- **Add your own dataset:** Edit `trainingData.json` to add new tasks.
- **Change network options:** Modify activation functions, layer sizes, etc. via the UI or JSON.

## Dependencies

- [p5.js](https://p5js.org/) (CDN)
- [Tailwind CSS](https://tailwindcss.com/) (CDN)
- Python (for local server)

## License

This project is open source and available under the MIT License.

## Author

Developed by [Sayantan](https://github.com/lazy-coder-03).

---

Feel free to contribute, suggest improvements, or fork for your own experiments!
