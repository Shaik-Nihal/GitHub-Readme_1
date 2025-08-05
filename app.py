import logging
import time
import os
from flask import Flask, render_template, request, jsonify
from readme_generator import (
    analyze_repo,
    generate_readme_from_analysis,
    get_repo_contents,
    parse_github_url,
    load_config,
    get_ai_readme_completion,
)
import github
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- In-Memory Cache ---
# A simple cache to store analysis results for a short period.
# In a production environment, a more robust solution like Redis or Memcached would be better.
ANALYSIS_CACHE = {}
CACHE_EXPIRATION_SECONDS = 600  # 10 minutes

# --- Routes ---
@app.route('/')
def index():
    """Renders the main page with the input form."""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    API endpoint to perform repository analysis.
    Checks cache before performing a full analysis.
    """
    try:
        data = request.get_json()
        repo_url = data.get('repo_url')
        github_token = data.get('github_token') or os.getenv('GITHUB_TOKEN')
        
        if not repo_url:
            return jsonify({"error": "repo_url is required."}), 400
        if not github_token:
            return jsonify({"error": "GitHub token is required. Set GITHUB_TOKEN in .env or provide in request."}), 400

        # Check cache first
        current_time = time.time()
        if repo_url in ANALYSIS_CACHE and (current_time - ANALYSIS_CACHE[repo_url]['timestamp'] < CACHE_EXPIRATION_SECONDS):
            app.logger.info(f"Cache hit for URL: {repo_url}")
            return jsonify(ANALYSIS_CACHE[repo_url]['data'])

        app.logger.info(f"Cache miss for URL: {repo_url}. Performing full analysis.")
        config = load_config()
        owner, repo_name = parse_github_url(repo_url)
        g = github.Github(github_token)
        repo = g.get_repo(f"{owner}/{repo_name}")
        repo_contents = get_repo_contents(repo, config)

        analysis_results = analyze_repo(repo_contents)

        response_data = {
            "analysis": analysis_results,
            "repo_contents": repo_contents
        }

        # Store in cache
        ANALYSIS_CACHE[repo_url] = {
            'timestamp': current_time,
            'data': response_data
        }

        return jsonify(response_data)

    except (ValueError, github.GithubException) as e:
        app.logger.warning(f"Analysis failed for URL '{repo_url if 'repo_url' in locals() else 'unknown'}': {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.critical(f"An unexpected critical error occurred during analysis: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred during analysis."}), 500

@app.route('/api/generate_full', methods=['POST'])
def generate_full():
    """
    API endpoint to generate the full README from analysis data.
    """
    try:
        data = request.get_json()
        analysis = data.get('analysis')
        repo_contents = data.get('repo_contents')
        ai_provider = data.get('ai_provider', 'openrouter')
        ai_model = data.get('ai_model', 'meta-llama/llama-3.2-3b-instruct:free')
        
        # Get API key based on provider
        if ai_provider == 'openrouter':
            api_key = os.getenv('OPENROUTER_API_KEY')
        elif ai_provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
        elif ai_provider == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')
        elif ai_provider == 'google':
            api_key = os.getenv('GOOGLE_API_KEY')
        else:
            api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
        
        if not analysis or not repo_contents:
            return jsonify({"error": "analysis and repo_contents are required."}), 400
        if not api_key:
            return jsonify({"error": f"API key for {ai_provider} is not configured in .env file."}), 400

        config = load_config()
        # Add AI provider and model to config
        config['ai_provider'] = ai_provider
        config['ai_model'] = ai_model
        
        readme = generate_readme_from_analysis(analysis, repo_contents, api_key, config)
        return jsonify({"readme": readme})

    except RuntimeError as e:
        app.logger.error(f"AI generation failed: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        app.logger.critical(f"An unexpected critical error occurred during full generation: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred during README generation."}), 500

@app.route('/api/regenerate_section', methods=['POST'])
def regenerate_section():
    """
    API endpoint to regenerate a single section of the README.
    """
    try:
        data = request.get_json()
        analysis = data.get('analysis')
        section_heading = data.get('section_heading')
        section_content = data.get('section_content')
        ai_provider = data.get('ai_provider', 'openrouter')
        ai_model = data.get('ai_model', 'meta-llama/llama-3.2-3b-instruct:free')
        
        # Get API key based on provider
        if ai_provider == 'openrouter':
            api_key = os.getenv('OPENROUTER_API_KEY')
        elif ai_provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
        elif ai_provider == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')
        elif ai_provider == 'google':
            api_key = os.getenv('GOOGLE_API_KEY')
        else:
            api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
        
        if not analysis or not section_heading:
            return jsonify({"error": "analysis and section_heading are required."}), 400
        if not api_key:
            return jsonify({"error": f"API key for {ai_provider} is not configured in .env file."}), 400

        config = load_config()
        # Add AI provider and model to config
        config['ai_provider'] = ai_provider
        config['ai_model'] = ai_model

        # Create a specific prompt for regenerating a section
        prompt = f"""
You are an expert technical writer. Based on the following project analysis, please regenerate the content for the README section titled '{section_heading}'.
The previous content was:
---
{section_content}
---

PROJECT ANALYSIS:
{json.dumps(analysis, indent=2)}

Please provide only the new content for this section, without the heading itself.
"""

        regenerated_content = get_ai_readme_completion(prompt, api_key, config)
        return jsonify({"content": regenerated_content})

    except RuntimeError as e:
        app.logger.error(f"Section regeneration failed: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        app.logger.critical(f"An unexpected critical error occurred during section regeneration: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred during section regeneration."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
