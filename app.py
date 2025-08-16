import logging
import time
import os
import re
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
            api_key = os.getenv('GOOGLE_AI_API_KEY')
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
        error_message = str(e)
        app.logger.error(f"AI generation failed: {error_message}")
        
        # Check for rate limit indicators
        if "429" in error_message or "rate limit" in error_message.lower():
            # Return a 200 response with a warning message instead of an error
            return jsonify({
                "readme": "# README Generation Started\n\nA rate limit was encountered while generating your README. The content is still being processed and will appear shortly. If it doesn't appear, try:\n\n1. Using a different AI model from the dropdown\n2. Waiting a minute and trying again\n3. Generating sections individually",
                "warning": "API rate limit encountered. The README was still generated but might be incomplete."
            }), 200
        else:
            return jsonify({"error": error_message}), 500
    except Exception as e:
        app.logger.critical(f"An unexpected critical error occurred during full generation: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred during README generation."}), 500

@app.route('/api/debug/context', methods=['GET'])
def debug_context():
    """
    API endpoint to debug the context stored for section regeneration.
    """
    try:
        from readme_generator import LATEST_FULL_PROMPT_CONTEXT_OVERVIEW
        context_status = {
            "is_available": bool(LATEST_FULL_PROMPT_CONTEXT_OVERVIEW),
            "type": str(type(LATEST_FULL_PROMPT_CONTEXT_OVERVIEW)),
            "keys": list(LATEST_FULL_PROMPT_CONTEXT_OVERVIEW.keys()) if isinstance(LATEST_FULL_PROMPT_CONTEXT_OVERVIEW, dict) else [],
            "summary_parts_count": len(LATEST_FULL_PROMPT_CONTEXT_OVERVIEW.get("summary_parts", [])) if isinstance(LATEST_FULL_PROMPT_CONTEXT_OVERVIEW, dict) else 0,
            "insights_parts_count": len(LATEST_FULL_PROMPT_CONTEXT_OVERVIEW.get("insights_parts", [])) if isinstance(LATEST_FULL_PROMPT_CONTEXT_OVERVIEW, dict) else 0,
            "file_contents_count": len(LATEST_FULL_PROMPT_CONTEXT_OVERVIEW.get("file_contents", [])) if isinstance(LATEST_FULL_PROMPT_CONTEXT_OVERVIEW, dict) else 0
        }
        return jsonify(context_status)
    except Exception as e:
        app.logger.error(f"Error in debug_context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/regenerate_section', methods=['POST'])
def regenerate_section():
    """
    API endpoint to regenerate a single section of the README.
    """
    try:
        from readme_generator import LATEST_FULL_PROMPT_CONTEXT_OVERVIEW, get_ai_readme_completion
        
        data = request.get_json()
        if not data:
            app.logger.error("No JSON data received in request")
            return jsonify({"error": "No JSON data received"}), 400
        
        app.logger.info(f"Regenerate section request received: {json.dumps(data)}")
            
        analysis = data.get('analysis')  # This is now optional
        section_heading = data.get('section_heading')
        section_content = data.get('section_content', '')
        ai_provider = data.get('ai_provider', 'openrouter')
        ai_model = data.get('ai_model', 'meta-llama/llama-3.2-3b-instruct:free')
        
        app.logger.info(f"Parsed parameters: heading='{section_heading}', content_length={len(section_content) if section_content else 0}, provider={ai_provider}, model={ai_model}")
        
        app.logger.info(f"Regenerating section: {section_heading} using {ai_provider}/{ai_model}")
        
        # Get API key based on provider
        if ai_provider == 'openrouter':
            api_key = os.getenv('OPENROUTER_API_KEY')
        elif ai_provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
        elif ai_provider == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')
        elif ai_provider == 'google':
            api_key = os.getenv('GOOGLE_AI_API_KEY')
        else:
            api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
        
        # Enhanced validation with detailed logging
        if not section_heading:
            app.logger.error("Missing required parameter: section_heading")
            return jsonify({"error": "section_heading is required."}), 400
            
        # Make section_content optional - provide default if not provided
        if not section_content:
            app.logger.warning("No section_content provided, using default")
            section_content = f"## {section_heading}\n\nPlease generate content for the {section_heading} section based on the repository analysis."
            
        if not api_key:
            app.logger.error(f"Missing API key for provider: {ai_provider}")
            return jsonify({"error": f"API key for {ai_provider} is not configured in .env file."}), 400
            
        # Check if the global context is available, but don't require it
        if not LATEST_FULL_PROMPT_CONTEXT_OVERVIEW:
            app.logger.warning("No context available for regeneration. Using basic regeneration without full context.")
            # Continue with regeneration using just the provided data

        config = load_config()
        # Add AI provider and model to config
        config['ai_provider'] = ai_provider
        config['ai_model'] = ai_model

        # Create a more comprehensive prompt using saved context if available
        context_info = ""
        if LATEST_FULL_PROMPT_CONTEXT_OVERVIEW and isinstance(LATEST_FULL_PROMPT_CONTEXT_OVERVIEW, dict):
            summary_parts = LATEST_FULL_PROMPT_CONTEXT_OVERVIEW.get("summary_parts", [])
            insights_parts = LATEST_FULL_PROMPT_CONTEXT_OVERVIEW.get("insights_parts", [])
            
            if summary_parts:
                context_info += "**PROJECT EXECUTIVE SUMMARY:**\n" + "\n".join(summary_parts) + "\n\n"
            if insights_parts:
                context_info += "**ACTIONABLE INSIGHTS:**\n" + "\n".join(insights_parts) + "\n\n"
                
            # Add relevant file snippets (limit to 3-5 to avoid token limits)
            file_contents = LATEST_FULL_PROMPT_CONTEXT_OVERVIEW.get("file_contents", [])
            if file_contents:
                context_info += "**KEY FILE SNIPPETS:**\n" + "\n".join(file_contents[:3]) + "\n\n"
        else:
            # Fallback to the provided analysis if no context is saved
            context_info = f"PROJECT ANALYSIS:\n{json.dumps(analysis, indent=2)}\n\n" if analysis else "No analysis data available. "
            app.logger.warning("No LATEST_FULL_PROMPT_CONTEXT_OVERVIEW available for section regeneration.")
        
        # Create a specific prompt for regenerating a section
        prompt = f"""
You are an expert technical writer specializing in GitHub README documentation. Your task is to regenerate ONLY the content for the README section titled '{section_heading}'.

{context_info}

The previous content for the '{section_heading}' section was:
---
{section_content}
---

INSTRUCTIONS:
1. Create an improved version of ONLY this section.
2. Use visual elements like tables, code blocks, emojis, and badges where appropriate.
3. Be comprehensive but clear and concise.
4. Follow Markdown best practices with proper formatting.
5. Do NOT include the section heading (like '## {section_heading}') in your response.
6. Start your content immediately without any introductions.
7. For the '{section_heading}' section specifically, focus on:
   - Providing clear and specific information
   - Using bullet points and lists where appropriate
   - Including code examples if relevant
   - Making it visually appealing with Markdown formatting

IMPORTANT: Return ONLY the content that should go under this section heading, without the heading itself.
"""

        # Generate the new section content
        regenerated_content = get_ai_readme_completion(prompt, api_key, config)
        
        # Clean up the content if needed (remove any accidental section headings the AI might add)
        cleaned_content = regenerated_content
        
        # Remove any leading/trailing # headings that might have been added by mistake
        cleaned_content = re.sub(r'^#+\s+.*?\n', '', cleaned_content)
        cleaned_content = re.sub(r'\n#+\s+.*?$', '', cleaned_content)
        
        return jsonify({"content": cleaned_content})

    except RuntimeError as e:
        error_message = str(e)
        app.logger.error(f"Section regeneration failed: {error_message}")
        
        # Check for rate limit indicators
        if "429" in error_message or "rate limit" in error_message.lower():
            # Return content with a warning about rate limits
            return jsonify({
                "content": "**Note: Rate limit encountered**\n\nThis section couldn't be regenerated due to API rate limits. Please try again in a minute or try with a different AI model. The original content remains unchanged.",
                "warning": "API rate limit encountered. Please try again in 60 seconds."
            }), 200
        else:
            return jsonify({"error": error_message}), 500
            
    except Exception as e:
        app.logger.critical(f"An unexpected critical error occurred during section regeneration: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred during section regeneration."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
    
