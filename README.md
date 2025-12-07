# KnowledgeLens AI: Extract & Visualize Insights from Any Document 🧠

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-live-demo-link-here.com) <!-- TODO: Replace with your actual live demo link if you host it! -->

Turn your unstructured documents (PDFs, text files, code, XML) into powerful, interactive knowledge graphs using the magic of Large Language Models (LLMs)! KnowledgeLens AI helps you understand complex information, discover hidden connections, and chat with your data.

## ✨ Features

*   **📄 Versatile Document Ingestion:** Upload PDFs, Markdown, TXT, code files (.py, .js, .java, etc.), JSON, XML, and more.
*   **💡 LLM-Powered Extraction:** Leverages OpenAI-compatible LLMs (Ollama, Llama.cpp, OpenAI API) to automatically identify and extract entities, concepts, and relationships.
*   **🗺️ Interactive Knowledge Graph:** Visualize your data as a dynamic, zoomable, and draggable graph powered by `pyvis` and `networkx`.
*   **🎯 Master Concept Detection:** Automatically detects the central theme of your uploaded documents or lets you set it manually.
*   **💬 Graph-Augmented Chat (RAG):** Ask questions about your graph, and get answers grounded *only* in the extracted relationships and entities.
*   **💾 Persistence:** Save and load your graph state to pick up where you left off without reprocessing.
*   **📤 Export Options:** Export your graph data in JSON or human-readable text formats.

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   Access to an OpenAI-compatible LLM endpoint (e.g., [Ollama](https://ollama.com/), [Llama.cpp](https://github.com/ggerganov/llama.cpp), or the OpenAI API).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Varun-SV/KnowledgeLens-AI.git
    cd KnowledgeLens-AI
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ./venv/Scripts/activate # On Windows
    # source venv/bin/activate # On macOS/Linux
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    
### Running the Application

1.  **Start your LLM endpoint:**
    *   If using Ollama, ensure it's running and you've pulled a model (e.g., `ollama run llama3.1`).
    *   If using `llama.cpp`, ensure your server is running.

2.  **Run the Streamlit application:**
    ```bash
    streamlit run KnowledgeLens_AI.py
    ```

    The application will open in your default web browser (usually `http://localhost:8501`).

### Configuration

Once the app is running:

1.  **LLM Configuration (Sidebar):**
    *   **Base URL:** Enter the URL of your LLM endpoint (e.g., `http://localhost:11434` for Ollama, `http://localhost:8080` for Llama.cpp, or `https://api.openai.com/v1` for OpenAI).
    *   **API Key (Optional):** Provide your API key if your endpoint requires authentication (e.g., OpenAI).
    *   **Model:** Specify the model name (e.g., `llama3.1`, `gpt-4o`).
    *   **Temperature:** Adjust for creativity vs. consistency (0.0 for factual, 1.0 for creative).

2.  **Graph Settings (Sidebar):**
    *   **Auto-detect master concept:** Let the AI find the main topic.
    *   **Manual master concept:** Set the main topic yourself.
    *   **Master connections (%):** Control how many top entities connect to the master node.
    *   **Custom Extraction Focus:** Provide specific instructions to the AI on what kind of relationships to prioritize during extraction.

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## ✉️ Contact

Your Name - [varunsv077@gmail.com](mailto:varunsv077@gmail.com)
Project Link: [https://github.com/Varun-SV/KnowledgeLens-AI](https://github.com/Varun-SV/KnowledgeLens-AI) <!-- TODO: Replace with your actual GitHub repo link -->
