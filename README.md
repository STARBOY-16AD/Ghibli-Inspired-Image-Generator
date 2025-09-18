Ghibli-Inspired Image Generator
Overview
The Ghibli-Inspired Image Generator is a web-based application that creates images inspired by the iconic art style of Studio Ghibli. Using Stable Diffusion models, either locally or through external APIs (Stability AI or Replicate), this tool generates whimsical, hand-drawn-style artwork with detailed backgrounds and natural lighting. The project features a Flask-based web interface, allowing users to input prompts, generate images, and view their generation history.
Features

Ghibli-Style Image Generation: Creates images mimicking Studio Ghibli's aesthetic, including fantasy landscapes, magical elements, and vibrant colors.
Flexible Model Options:
Use local Stable Diffusion model (runwayml/stable-diffusion-v1-5) with GPU support for faster inference.
Leverage external APIs (Stability AI or Replicate) for cloud-based generation.


Web Interface: Flask-powered UI with prompt input, sample prompts, image display, and user history tracking.
Prompt Enhancement: Automatically enhances user prompts with Ghibli-specific keywords for better results.
Generation History: Stores user-generated images and prompts in session-based memory (in-memory storage, not persistent).
Cross-Platform: Runs on systems with Python and required dependencies, with optional GPU acceleration.

Installation
Prerequisites

Python: 3.8 or higher
pip: Python package manager
Git: For cloning the repository
Optional:
NVIDIA GPU with CUDA support for local model inference
API keys for Stability AI or Replicate (required for cloud-based generation)
gunicorn (for production deployment)



Steps

Clone the Repository:
git clone https://github.com/STARBOY-16AD/Ghibli-Inspired-Image-Generator.git
cd Ghibli-Inspired-Image-Generator


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Configure Environment Variables:

Copy .env.template to .env:cp .env.template .env


Edit .env to include your API keys (for Stability AI or Replicate) or enable local model:STABILITY_API_KEY=your_stability_api_key_here
# or
REPLICATE_API_TOKEN=your_replicate_api_token_here
USE_LOCAL_MODEL=false
FLASK_DEBUG=false
SECRET_KEY=your_random_secret_key_here
PORT=5000


For local model usage, set USE_LOCAL_MODEL=true and ensure GPU drivers and PyTorch with CUDA are installed.


Run the Application:
python frontend.py

Access the web interface at http://localhost:5000.


Usage

Access the Web Interface:

Open http://localhost:5000 in your browser.
Enter a prompt (e.g., "A peaceful forest with spirits hiding among the trees") or select a sample prompt.
Click "Generate" to create a Ghibli-style image.


View History:

Generated images and prompts are saved in your session history, accessible via the "History" section.
Clear history using the "Clear History" button.


Run Locally with Command Line (for testing):
python backend.py

This generates a test image with the prompt "A peaceful forest with small spirits hiding among trees and flowers".


Project Structure
Ghibli-Inspired-Image-Generator/
├── .env                   # Environment variables (not tracked)
├── .env.template          # Template for environment variables
├── .gitignore             # Git ignore file
├── backend.py             # Core image generation logic (Stable Diffusion, API integration)
├── frontend.py            # Flask web interface
├── ghibli_generator.py    # Bridge between frontend and backend
├── huggingface_compat.py  # Compatibility layer for Hugging Face Hub
├── requirements.txt       # Python dependencies
├── static/                # Static files (generated images)
│   └── generated/         # Directory for generated images
└── templates/             # Flask templates (index.html)

Configuration

Local Model:
Requires USE_LOCAL_MODEL=true in .env.
Uses runwayml/stable-diffusion-v1-5 with FP16 precision and DPM solver for faster inference.
Automatically moves to GPU if available.


API-Based Generation:
Requires STABILITY_API_KEY or REPLICATE_API_TOKEN in .env.
Uses Stability AI's SDXL or Replicate's SDXL model with Ghibli-specific prompts.


Flask Settings:
FLASK_DEBUG: Enable debug mode (set to false in production).
SECRET_KEY: Random string for session security.
PORT: Port for the web server (default: 5000).



Dependencies

flask: Web framework for the interface.
python-dotenv: Loads environment variables.
torch: PyTorch for local model inference.
diffusers: Hugging Face Diffusers for Stable Diffusion.
huggingface-hub: Model downloading and caching.
requests: API calls to Stability AI/Replicate.
Pillow: Image processing.
gunicorn: WSGI server for production.

See requirements.txt for full list.
Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch:git checkout -b feature/your-feature-name


Commit your changes:git commit -m "Add your feature description"


Push to your fork:git push origin feature/your-feature-name


Open a pull request with a clear description of your changes.

Please include tests and follow Python PEP 8 style guidelines.
Troubleshooting

Model Loading Errors: Ensure compatible PyTorch and Hugging Face versions. Try clearing the Hugging Face cache (~/.cache/huggingface).
API Errors: Verify API keys in .env and check Stability AI/Replicate documentation for rate limits or model availability.
GPU Issues: Confirm CUDA is installed and compatible with your PyTorch version.
Web Interface Issues: Set FLASK_DEBUG=true to see detailed errors.

License
This project is licensed under the MIT License. See the LICENSE file for details (if added).
Acknowledgments

Built with Stable Diffusion, Hugging Face Diffusers, and APIs from Stability AI and Replicate.
Inspired by the artistry of Studio Ghibli.
Thanks to the open-source community for their tools and libraries.

Contact
For issues or suggestions, open an issue on the GitHub repository.
Create your own Ghibli-inspired artwork and bring your imagination to life!
