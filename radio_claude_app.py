import gradio as gr
import anthropic
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from typing import List, Dict, Any
import traceback

# Global variables
client = None
conversation_history = []

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)

def set_api_key(api_key: str) -> str:
    """Set the Anthropic API key"""
    global client
    if not api_key.strip():
        return "❌ Please enter a valid API key"
    
    try:
        client = anthropic.Anthropic(api_key=api_key.strip())
        # Test the API key with a simple request
        test_response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        return "✅ API key set successfully!"
    except Exception as e:
        client = None
        return f"❌ Error setting API key: {str(e)}"

def train_mnist_logistic_regression() -> str:
    """Tool to train a logistic regression model on MNIST dataset"""
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load MNIST dataset (using a subset for faster training)
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        # Use subset for faster training in demo
        subset_size = 10000
        train_subset = torch.utils.data.Subset(train_dataset, range(subset_size))
        
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        # Initialize model
        model = LogisticRegression(28*28, 10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Training
        model.train()
        epoch_losses = []
        
        for epoch in range(5):  # Reduced epochs for demo
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                data = data.view(-1, 28*28)  # Flatten
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            
            epoch_loss = running_loss / len(train_loader)
            epoch_losses.append(epoch_loss)
        
        # Testing
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(-1, 28*28)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        
        # Save model
        torch.save(model.state_dict(), 'mnist_logistic_model.pth')
        
        # Create training plot
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epoch_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Show some predictions
        plt.subplot(1, 2, 2)
        model.eval()
        with torch.no_grad():
            sample_data, sample_targets = next(iter(test_loader))
            sample_data = sample_data[:6].to(device)
            sample_targets = sample_targets[:6]
            sample_data_flat = sample_data.view(-1, 28*28)
            sample_outputs = model(sample_data_flat)
            _, sample_predictions = torch.max(sample_outputs, 1)
            
            for i in range(6):
                plt.subplot(2, 3, i+1)
                plt.imshow(sample_data[i].cpu().squeeze(), cmap='gray')
                plt.title(f'True: {sample_targets[i]}, Pred: {sample_predictions[i].cpu()}')
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('mnist_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        result = f"""
🎯 **MNIST Logistic Regression Training Complete!**

**Results:**
- Final Training Loss: {epoch_losses[-1]:.4f}
- Test Accuracy: {accuracy:.2f}%
- Total Test Samples: {total:,}
- Correct Predictions: {correct:,}
- Device Used: {device}

**Model Details:**
- Architecture: Logistic Regression (784 → 10)
- Optimizer: SGD (lr=0.01)
- Training Samples: {subset_size:,}
- Epochs: 5
- Batch Size: 64

**Files Generated:**
- `mnist_logistic_model.pth` - Trained model weights
- `mnist_results.png` - Training loss and sample predictions

The model achieves {accuracy:.2f}% accuracy on the MNIST test set!
        """
        
        return result.strip()
        
    except Exception as e:
        return f"❌ Error training model: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

def get_available_tools() -> List[Dict[str, Any]]:
    """Define available tools for the chatbot"""
    return [
        {
            "name": "train_mnist_logistic_regression",
            "description": "Train a logistic regression model on the MNIST handwritten digit dataset using PyTorch",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]

def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Execute a tool based on its name"""
    if tool_name == "train_mnist_logistic_regression":
        return train_mnist_logistic_regression()
    else:
        return f"❌ Unknown tool: {tool_name}"

def chat_with_claude(message: str, history: List[List[str]], api_key: str) -> tuple:
    """Chat with Claude and handle tool calls"""
    global client, conversation_history
    
    if not client:
        error_msg = "❌ Please set your API key first in the sidebar"
        history.append([message, error_msg])
        return history, ""
    
    try:
        # Add user message to conversation history
        conversation_history.append({"role": "user", "content": message})
        
        # Create the system message with tool definitions
        system_message = """You are a helpful AI assistant with access to machine learning tools. 

You can help users with:
1. General conversation and questions
2. Machine learning and data science topics  
3. Using tools to train models

Available tools:
- train_mnist_logistic_regression: Train a logistic regression model on MNIST dataset

When a user asks about training MNIST models or logistic regression, you can use the tool to actually train the model and provide real results.

Be helpful, informative, and use tools when appropriate to provide hands-on demonstrations."""

        # Make API call to Claude
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            system=system_message,
            tools=get_available_tools(),
            messages=conversation_history
        )
        
        assistant_response = ""
        
        # Process the response
        for content_block in response.content:
            if content_block.type == "text":
                assistant_response += content_block.text
            elif content_block.type == "tool_use":
                tool_name = content_block.name
                tool_input = content_block.input
                
                assistant_response += f"\n\n🔧 **Using tool: {tool_name}**\n\n"
                
                # Execute the tool
                tool_result = execute_tool(tool_name, tool_input)
                assistant_response += tool_result
                
                # Add tool result to conversation history
                conversation_history.append({
                    "role": "assistant", 
                    "content": [
                        {"type": "tool_use", "id": content_block.id, "name": tool_name, "input": tool_input}
                    ]
                })
                conversation_history.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": content_block.id, "content": tool_result}
                    ]
                })
        
        # Add assistant response to conversation history
        if assistant_response and not any(block.type == "tool_use" for block in response.content):
            conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # Add to gradio history
        history.append([message, assistant_response])
        
        return history, ""
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history.append([message, error_msg])
        return history, ""

def clear_conversation():
    """Clear the conversation history"""
    global conversation_history
    conversation_history = []
    return []

def create_app():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Claude Chat with ML Tools", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🤖 Claude Chat Assistant with ML Tools
        
        This app provides an AI chat interface powered by Anthropic's Claude with integrated machine learning tools.
        
        **Features:**
        - 💬 Chat with Claude AI
        - 🛠️ Use ML tools via chat commands  
        - 🧠 Train logistic regression on MNIST dataset
        """)
        
        with gr.Row():
            # Main chat area
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Chat with Claude",
                    height=500,
                    placeholder="Start chatting with Claude! Try asking: 'Can you train a logistic regression model on MNIST?'"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            # Sidebar
            with gr.Column(scale=1):
                gr.Markdown("### 🔑 Configuration")
                
                api_key_input = gr.Textbox(
                    label="Anthropic API Key",
                    type="password",
                    placeholder="sk-ant-api03-...",
                    info="Enter your Anthropic API key to enable chat"
                )
                
                api_key_btn = gr.Button("Set API Key", variant="primary")
                api_key_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="Enter API key above"
                )
                
                gr.Markdown("### 🛠️ Available Tools")
                gr.Markdown("""
                **Machine Learning Tools:**
                - `train_mnist_logistic_regression` - Train a PyTorch logistic regression model on MNIST
                
                **Usage Examples:**
                - "Train a logistic regression model on MNIST"
                - "Can you build an MNIST classifier?"
                - "Show me how to train a model on handwritten digits"
                """)
                
                gr.Markdown("### ℹ️ Instructions")
                gr.Markdown("""
                1. **Set API Key**: Enter your Anthropic API key in the field above
                2. **Start Chatting**: Ask questions or request to use tools
                3. **Use Tools**: Ask Claude to train models or perform ML tasks
                4. **View Results**: See training results and model performance
                """)
        
        # Event handlers
        def send_message(message, history, api_key):
            if not message.strip():
                return history, ""
            return chat_with_claude(message, history, api_key)
        
        # Button click events
        send_btn.click(
            send_message,
            inputs=[msg, chatbot, api_key_input],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            send_message,
            inputs=[msg, chatbot, api_key_input], 
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            clear_conversation,
            outputs=[chatbot]
        )
        
        api_key_btn.click(
            set_api_key,
            inputs=[api_key_input],
            outputs=[api_key_status]
        )
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
