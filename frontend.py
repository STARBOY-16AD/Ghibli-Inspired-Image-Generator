from flask import Flask, render_template, request, jsonify, url_for, redirect, session
import os
import secrets
from datetime import datetime
from ghibli_generator import GhibliArtGenerator

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(16))
app.config['UPLOAD_FOLDER'] = 'static/generated'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the generator
generator = GhibliArtGenerator()

# In-memory storage for history (replace with database in production)
if not hasattr(app, 'generation_history'):
    app.generation_history = {}

@app.route('/')
def index():
    """Render the main page"""
    # Get user session ID or create a new one
    user_id = session.get('user_id')
    if not user_id:
        user_id = secrets.token_hex(16)
        session['user_id'] = user_id
        app.generation_history[user_id] = []
    
    # Get user history
    history = app.generation_history.get(user_id, [])
    
    # Get sample prompts
    sample_prompts = [
        "A peaceful forest with spirits hiding among the trees",
        "Flying castle floating in the clouds at sunset",
        "Cat bus driving through a mystical tunnel",
        "Countryside train ride by the ocean",
        "Magical bathhouse with lanterns at night",
        "Young witch flying on a broomstick over a city",
        "Giant forest spirit walking through trees",
        "River dragon soaring through the sky"
    ]
    
    return render_template('index.html', history=history, sample_prompts=sample_prompts)

@app.route('/generate', methods=['POST'])
def generate_art():
    """Generate art based on prompt"""
    prompt = request.form.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    try:
        # Generate the image
        image_path = generator.generate(prompt)
        
        # Get relative path for URL
        relative_path = image_path.replace('\\', '/')
        if relative_path.startswith('static/'):
            relative_path = relative_path[7:]  # Remove 'static/' prefix
        
        image_url = url_for('static', filename=relative_path)
        
        # Store in history
        user_id = session.get('user_id')
        if user_id:
            if user_id not in app.generation_history:
                app.generation_history[user_id] = []
            
            app.generation_history[user_id].append({
                'prompt': prompt,
                'image_url': image_url,
                'timestamp': datetime.now().isoformat()
            })
        
        return jsonify({'success': True, 'image_url': image_url})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    """Show generation history"""
    user_id = session.get('user_id')
    if not user_id or user_id not in app.generation_history:
        return jsonify([])
    
    return jsonify(app.generation_history[user_id])

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear generation history"""
    user_id = session.get('user_id')
    if user_id and user_id in app.generation_history:
        app.generation_history[user_id] = []
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    # In production, use a proper WSGI server
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv("FLASK_DEBUG", "False").lower() == "true")