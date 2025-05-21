#!/bin/bash

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed"
fi

# Pull recommended models
echo "Pulling recommended models..."
ollama pull mistral:7b-instruct
ollama pull llama2:13b

echo "Setup complete! You can now run the application with:"
echo "npm run dev" 