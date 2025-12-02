[README.md](https://github.com/user-attachments/files/23869344/README.md)
# 🤖 Claude Chat Assistant with ML Tools

A powerful Gradio web application that combines Anthropic's Claude AI with integrated machine learning tools. Chat with Claude naturally and seamlessly execute ML tasks like training models on MNIST dataset using PyTorch.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Gradio](https://img.shields.io/badge/gradio-4.0+-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-green.svg)

## 🌟 Features

### 💬 **Claude AI Integration**
- Real-time chat interface powered by Anthropic's Claude
- Maintains conversation history throughout the session
- Secure API key management with validation
- Professional, user-friendly interface

### 🛠️ **Machine Learning Tools**
- **Normal Chat Mode**: Have regular conversations with Claude
- **Tool Calling Mode**: Claude automatically calls ML tools when requested
- **MNIST Logistic Regression**: Train PyTorch models on handwritten digits
- **Extensible Architecture**: Easy to add more ML tools

### 🎯 **MNIST Classifier Tool**
- Complete PyTorch logistic regression implementation
- Automatic dataset download and preprocessing
- Real-time training with progress monitoring
- Model evaluation with accuracy metrics
- Visualization of training loss and sample predictions
- Model persistence (saves trained weights)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Anthropic API key ([Get one here](https://console.anthropic.com/))

### Installation

1. **Clone or download the files:**
   ```bash
   # Download the main files:
   # - gradio_claude_app.py
   # - requirements.txt
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python gradio_claude_app.py
   ```

4. **Open in your browser:**
   ```
   http://localhost:7860
   ```

## 🔧 Configuration

### API Key Setup
1. Get your Anthropic API key from [console.anthropic.com](https://console.anthropic.com/)
2. Enter the API key in the sidebar of the application
3. Click "Set API Key" to validate and activate

### Environment Variables (Optional)
You can also set your API key as an environment variable:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## 💡 Usage Examples

### Normal Conversation
Just chat naturally with Claude:
```
User: "Hi Claude, how are you today?"
User: "Can you explain what machine learning is?"
User: "What's the difference between classification and regression?"
```

### ML Tool Usage
Ask Claude to use the integrated tools:
```
User: "Can you train a logistic regression model on MNIST?"
User: "Build a handwritten digit classifier for me"
User: "Train a neural network on the MNIST dataset"
User: "Show me how to classify handwritten digits"
```

### Advanced Queries
Combine conversation with tool usage:
```
User: "First explain logistic regression, then train one on MNIST"
User: "What is MNIST dataset and can you train a model on it?"
User: "Compare different ML algorithms, then demonstrate with MNIST"
```

## 🏗️ Architecture

### Core Components

```
gradio_claude_app.py
├── API Key Management
├── Claude Chat Interface
├── Tool Integration System
└── ML Tools
    └── MNIST Logistic Regression
```

### Tool System
The application uses a flexible tool system where Claude can automatically call functions:

```python
# Tool Definition
{
    "name": "train_mnist_logistic_regression",
    "description": "Train a logistic regression model on MNIST dataset",
    "input_schema": {...}
}

# Automatic Execution
Claude detects user intent → Calls appropriate tool → Returns results
```

### MNIST Implementation Details

**Model Architecture:**
- Input: 784 features (28×28 flattened images)
- Output: 10 classes (digits 0-9)
- Architecture: Linear layer with softmax

**Training Configuration:**
- Optimizer: SGD (learning rate: 0.01)
- Loss Function: CrossEntropyLoss
- Epochs: 5 (configurable)
- Batch Size: 64
- Dataset: Subset of 10,000 training samples (for demo speed)

**Outputs Generated:**
- `mnist_logistic_model.pth` - Trained model weights
- `mnist_results.png` - Training visualizations
- Detailed performance metrics in chat

## 📊 Example Output

When you ask Claude to train a model, you'll get comprehensive results:

```
🎯 MNIST Logistic Regression Training Complete!

Results:
- Final Training Loss: 0.5234
- Test Accuracy: 91.23%
- Total Test Samples: 10,000
- Correct Predictions: 9,123
- Device Used: cpu

Model Details:
- Architecture: Logistic Regression (784 → 10)
- Optimizer: SGD (lr=0.01)
- Training Samples: 10,000
- Epochs: 5
- Batch Size: 64

Files Generated:
- mnist_logistic_model.pth - Trained model weights
- mnist_results.png - Training loss and sample predictions
```

## 🔧 Customization

### Adding New ML Tools

1. **Define the tool function:**
   ```python
   def your_new_tool() -> str:
       """Your ML tool implementation"""
       try:
           # Your ML code here
           return "Success message with results"
       except Exception as e:
           return f"Error: {str(e)}"
   ```

2. **Add tool definition:**
   ```python
   {
       "name": "your_new_tool",
       "description": "Description of what your tool does",
       "input_schema": {...}
   }
   ```

3. **Register in execute_tool():**
   ```python
   def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
       if tool_name == "your_new_tool":
           return your_new_tool()
   ```

### Modifying MNIST Training

Edit the `train_mnist_logistic_regression()` function to customize:
- Number of epochs
- Learning rate
- Batch size
- Model architecture
- Dataset size

## 📋 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| gradio | ≥4.0.0 | Web interface framework |
| anthropic | ≥0.7.0 | Claude API client |
| torch | ≥2.0.0 | Deep learning framework |
| torchvision | ≥0.15.0 | Computer vision utilities |
| matplotlib | ≥3.5.0 | Plotting and visualization |
| numpy | ≥1.21.0 | Numerical computing |
| Pillow | ≥8.3.0 | Image processing |

## 🚨 Troubleshooting

### Common Issues

**API Key Error:**
```
❌ Error setting API key: Invalid API key
```
- Solution: Check your API key is correct and active
- Verify it starts with 'sk-ant-api03-'

**CUDA/GPU Issues:**
```
RuntimeError: CUDA out of memory
```
- Solution: The app automatically falls back to CPU
- Reduce batch size in the training function if needed

**Import Errors:**
```
ModuleNotFoundError: No module named 'torch'
```
- Solution: Run `pip install -r requirements.txt`
- Ensure you're using Python 3.8+

**Training Timeout:**
```
Training taking too long...
```
- The MNIST tool uses a subset (10K samples) for speed
- Training typically takes 1-3 minutes on CPU
- Check console for progress updates

### Performance Tips

1. **GPU Acceleration**: If you have CUDA available, the app will automatically use GPU
2. **Memory Management**: Large datasets are automatically subsampled for demo purposes
3. **Concurrent Users**: Each user session is independent with separate conversation history

## 📁 Project Structure

```
claude-ml-chat/
├── gradio_claude_app.py      # Main application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── data/                    # MNIST dataset (auto-downloaded)
├── mnist_logistic_model.pth # Saved model (generated)
└── mnist_results.png        # Training visualizations (generated)
```

## 🤝 Contributing

Want to add more ML tools or improve the interface? Here's how:

1. **Fork the project**
2. **Add your ML tool** following the pattern in `train_mnist_logistic_regression()`
3. **Update the tool registry** in `get_available_tools()`
4. **Test thoroughly** with various user inputs
5. **Submit a pull request**

### Ideas for New Tools
- 🌸 Iris classification with scikit-learn
- 🏠 Boston housing price prediction
- 📊 Time series forecasting
- 🖼️ Image classification with CNN
- 📝 Text classification with transformers

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

- This application requires an active Anthropic API key
- API usage is subject to Anthropic's pricing and rate limits
- ML training uses computational resources - monitor your usage
- Generated models are for demonstration purposes

## 🆘 Support

Having issues? Try these resources:

1. **Check the troubleshooting section** above
2. **Review the console output** for detailed error messages  
3. **Verify API key** is correctly set and has sufficient credits
4. **Check dependencies** are properly installed

## 🔄 Updates

### Version History
- **v1.0.0** - Initial release with Claude chat and MNIST tool
- **v1.1.0** - Enhanced error handling and visualization
- **v1.2.0** - Added model persistence and performance metrics

### Planned Features
- 🔄 More ML algorithms (Random Forest, SVM, Neural Networks)
- 📊 Advanced visualization tools
- 💾 Conversation export functionality  
- 🔧 Custom dataset upload
- 📱 Mobile-responsive design improvements

---

**Built with ❤️ using Gradio, Claude AI, and PyTorch**

*Happy machine learning! 🚀*
