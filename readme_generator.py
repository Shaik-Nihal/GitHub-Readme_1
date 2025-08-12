# Main script for the GitHub Readme Generator
import os
import requests
import github
from dotenv import load_dotenv
import yaml
import argparse
import logging
import datetime
import json
import re

# Load environment variables
load_dotenv()

# --- Global Logger ---
# It's better to configure the logger in the application entry point (main or app.py)
# but for standalone script use, we can have a default configuration.
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
DEFAULT_CONFIG = {
    "ai_model": "gpt-3.5-turbo",
    "ai_temperature": 0.7,
    "ai_max_tokens": 2048,
    "max_prompt_files": 15,
    "max_content_snippet_length": 600,
    "excluded_dirs": ["docs/", "examples/", "tests/", "test/", "data/", "samples/"],
    "excluded_files": ["*.log", "*.tmp", "*.swp", ".DS_Store"],
    "readme_sections": [
        "Project Title", "Description", "Key Features", "Technologies Used",
        "Installation", "Usage", "Contributing", "License"
    ],
    "readme_template_path": None,
    "max_individual_file_size_bytes": 1000000,
}

def load_config(config_path_override=None):
    # This function remains largely the same but will be called by the main generator function
    config = DEFAULT_CONFIG.copy()
    config_file_to_load = config_path_override if config_path_override else "config.yaml"
    if not os.path.exists(config_file_to_load):
        if config_path_override:
            logger.warning(f"Specified config file '{config_file_to_load}' not found. Using default configuration.")
        else:
            logger.info("config.yaml not found, using default configuration.")
        return config
    try:
        with open(config_file_to_load, "r") as f:
            user_config = yaml.safe_load(f)
        if user_config:
            config.update(user_config)
            logger.info(f"Loaded configuration from {config_file_to_load}")
    except Exception as e:
        logger.error(f"Error loading {config_file_to_load}: {e}. Using default configuration.", exc_info=True)
    return config

# This global variable will hold the latest prompt context overview for section regeneration.
# Initialize it as a dictionary with empty lists for the required keys
LATEST_FULL_PROMPT_CONTEXT_OVERVIEW = {
    "summary_parts": [],
    "insights_parts": [],
    "file_contents": []
}

def parse_github_url(url: str) -> tuple[str, str]:
    """Parses a GitHub URL to extract owner and repo name. Handles various formats."""
    processed_url = url.strip()
    if not re.match(r"^[a-zA-Z0-9]+://", processed_url) and not processed_url.lower().startswith("git@"):
        if re.search(r"github\.com/", processed_url, re.IGNORECASE):
            processed_url = "https://" + processed_url

    url_to_match = processed_url
    if processed_url.lower().startswith(("http://", "https://")):
        parts = processed_url.split('/')
        if len(parts) > 2:
            parts[0] = parts[0].lower()
            parts[2] = parts[2].lower()
            url_to_match = '/'.join(parts)
    elif processed_url.lower().startswith("git@"):
        parts = processed_url.split(':')
        if len(parts) > 1:
            domain_part = parts[0].split('@')
            if len(domain_part) > 1:
                domain_part[1] = domain_part[1].lower()
                parts[0] = '@'.join(domain_part)
            url_to_match = ':'.join(parts)

    patterns = [
        r"^(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
        r"^git@github\.com:([^/]+)/([^/]+?)(?:\.git)?/?$"
    ]
    for pattern in patterns:
        match = re.match(pattern, url_to_match)
        if match:
            owner, repo = match.groups()
            repo = repo.removesuffix('/').removesuffix('.')
            logger.debug(f"Parsed URL: owner='{owner}', repo='{repo}' from URL '{url}'")
            return owner, repo

    logger.error(f"Invalid or unsupported GitHub URL format: '{url}'")
    raise ValueError(f"Invalid or unsupported GitHub URL format: {url}")

def get_repo_contents(repo, config, current_path=""):
    # This function remains largely the same, but takes config as an argument
    repo_structure = {}
    default_excluded_dirs = {".git/", "node_modules/", "__pycache__/", "dist/", "build/", "target/", "vendor/", ".venv/", "env/"}
    config_excluded_dirs_raw = config.get("excluded_dirs", [])
    config_excluded_dirs = {d.strip().rstrip('/') + '/' for d in config_excluded_dirs_raw if d.strip()}
    all_excluded_dirs = default_excluded_dirs.union(config_excluded_dirs)
    config_excluded_files_patterns = config.get("excluded_files", [])

    try:
        contents = repo.get_contents(current_path)
        for content_file in contents:
            item_path = content_file.path
            if any((item_path + "/").startswith(excluded_dir) for excluded_dir in all_excluded_dirs):
                logger.debug(f"Skipping '{item_path}' as it's within an excluded directory.")
                continue

            if content_file.type == "dir":
                repo_structure.update(get_repo_contents(repo, config, item_path))
            else:
                is_excluded = any(
                    (p.startswith("*.") and item_path.endswith(p[1:])) or item_path.endswith(p) for p in config_excluded_files_patterns
                )
                if is_excluded:
                    logger.debug(f"Skipping excluded file: '{item_path}'")
                    continue

                if content_file.size == 0:
                    repo_structure[item_path] = ""
                    continue

                max_file_size = config.get("max_individual_file_size_bytes", 1000000)
                if content_file.size > max_file_size:
                    logger.info(f"Skipping large file: '{item_path}' (Size: {content_file.size} > Limit: {max_file_size})")
                    repo_structure[item_path] = f"# File content skipped (too large: {content_file.size} bytes)"
                    continue

                try:
                    repo_structure[item_path] = content_file.decoded_content.decode('utf-8', errors='replace')
                except Exception as e:
                    logger.warning(f"Error decoding file '{item_path}': {e}. Storing error message.")
                    repo_structure[item_path] = f"# Error decoding file: {e}"
    except (github.UnknownObjectException, github.GitignoreBlockedException) as e:
        logger.warning(f"Could not access path '{current_path}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching contents for '{current_path}': {e}", exc_info=True)
        # Re-raise as a more generic error for the web app to handle
        raise RuntimeError(f"Failed to fetch repository contents for path: {current_path}") from e

    return repo_structure

import json

# --- Analysis Helper Functions ---

def analyze_dependencies(repo_contents: dict) -> dict:
    """
    Analyzes repository contents for dependency files and extracts libraries.
    """
    dependencies = {"python": [], "javascript": []}
    logger.info("Analyzing dependency files...")

    # Python: requirements.txt
    if "requirements.txt" in repo_contents:
        try:
            lines = repo_contents["requirements.txt"].splitlines()
            # Simple parsing: ignore comments, empty lines, and options
            deps = [line.strip().split('==')[0] for line in lines if line.strip() and not line.startswith('#')]
            dependencies["python"].extend(deps)
            logger.info(f"Found {len(deps)} Python dependencies in requirements.txt")
        except Exception as e:
            logger.warning(f"Could not parse requirements.txt: {e}")

    # Python: pyproject.toml (basic parsing)
    if "pyproject.toml" in repo_contents:
        try:
            # This is a very basic TOML parser, real-world use would need a library
            content = repo_contents["pyproject.toml"]
            if "[tool.poetry.dependencies]" in content:
                deps_str = content.split("[tool.poetry.dependencies]")[1].split('[')[0]
                lines = [line.strip().split('=')[0].strip() for line in deps_str.splitlines() if line.strip() and not line.startswith('#')]
                dependencies["python"].extend(lines)
                logger.info(f"Found {len(lines)} Python dependencies in pyproject.toml")
        except Exception as e:
            logger.warning(f"Could not parse pyproject.toml: {e}")

    # JavaScript: package.json
    if "package.json" in repo_contents:
        try:
            data = json.loads(repo_contents["package.json"])
            deps = list(data.get("dependencies", {}).keys())
            dev_deps = list(data.get("devDependencies", {}).keys())
            dependencies["javascript"].extend(deps)
            dependencies["javascript"].extend(dev_deps)
            logger.info(f"Found {len(deps) + len(dev_deps)} JavaScript dependencies in package.json")
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse package.json: {e}")

    return dependencies

def analyze_entry_points(repo_contents: dict) -> dict:
    """
    Analyzes repository for potential entry points and run commands.
    """
    entry_points = {"run_commands": [], "main_files": []}
    logger.info("Analyzing for entry points and run commands...")

    # package.json scripts
    if "package.json" in repo_contents:
        try:
            data = json.loads(repo_contents["package.json"])
            scripts = data.get("scripts", {})
            if "start" in scripts:
                entry_points["run_commands"].append(f"npm start (runs: `{scripts['start']}`)")
            if "dev" in scripts:
                entry_points["run_commands"].append(f"npm run dev (runs: `{scripts['dev']}`)")
        except json.JSONDecodeError:
            pass # Already logged in dependency analysis

    # Procfile
    if "Procfile" in repo_contents:
        lines = repo_contents["Procfile"].splitlines()
        for line in lines:
            if ":" in line:
                entry_points["run_commands"].append(f"From Procfile: `{line.strip()}`")

    # Common main files
    for f in ["main.py", "app.py", "index.js", "server.js"]:
        if f in repo_contents:
            entry_points["main_files"].append(f)

    return entry_points

def analyze_configuration(repo_contents: dict) -> dict:
    """
    Analyzes for configuration files like .env.example.
    """
    config_analysis = {"required_env_vars": []}
    logger.info("Analyzing for configuration files...")

    if ".env.example" in repo_contents:
        try:
            lines = repo_contents[".env.example"].splitlines()
            vars = [line.split('=')[0] for line in lines if line.strip() and not line.startswith('#')]
            config_analysis["required_env_vars"].extend(vars)
            logger.info(f"Found {len(vars)} environment variables in .env.example")
        except Exception as e:
            logger.warning(f"Could not parse .env.example: {e}")

    return config_analysis

def analyze_docker(repo_contents: dict) -> dict:
    """
    Performs a basic analysis of a Dockerfile.
    """
    docker_analysis = {"base_image": None, "exposed_ports": []}
    if "Dockerfile" in repo_contents:
        logger.info("Analyzing Dockerfile...")
        try:
            lines = repo_contents["Dockerfile"].splitlines()
            for line in lines:
                if line.strip().upper().startswith("FROM"):
                    docker_analysis["base_image"] = line.strip().split()[1]
                if line.strip().upper().startswith("EXPOSE"):
                    docker_analysis["exposed_ports"].append(line.strip().split()[1])
        except Exception as e:
            logger.warning(f"Could not parse Dockerfile: {e}")

    return docker_analysis

def analyze_project_metadata(repo_contents: dict) -> dict:
    """
    Analyzes for project metadata from files like pyproject.toml, package.json, and existing README.
    Enhanced to collect more details for README generation with badges, tables, etc.
    """
    metadata = {
        "project_name": None, 
        "description": None,
        "version": None,
        "author": None,
        "license": None,
        "repository_url": None,
        "homepage": None,
        "keywords": [],
        "badges": []
    }
    
    # Extract from pyproject.toml
    if "pyproject.toml" in repo_contents:
        logger.info("Analyzing pyproject.toml for metadata...")
        content = repo_contents["pyproject.toml"]
        
        # Basic extraction with simple string matching (not a full TOML parser)
        if 'name = "' in content:
            metadata["project_name"] = content.split('name = "')[1].split('"')[0]
        if 'description = "' in content:
            metadata["description"] = content.split('description = "')[1].split('"')[0]
        if 'version = "' in content:
            metadata["version"] = content.split('version = "')[1].split('"')[0]
        if 'license = "' in content:
            metadata["license"] = content.split('license = "')[1].split('"')[0]
        if 'authors = [' in content:
            authors_section = content.split('authors = [')[1].split(']')[0]
            if '"' in authors_section:
                metadata["author"] = authors_section.split('"')[1].split('"')[0]

    # Extract from package.json (with more comprehensive extraction)
    if "package.json" in repo_contents:
        try:
            data = json.loads(repo_contents["package.json"])
            
            # Basic metadata
            if not metadata["project_name"]:
                metadata["project_name"] = data.get("name")
            if not metadata["description"]:
                metadata["description"] = data.get("description")
            metadata["version"] = data.get("version")
            
            # Author information
            if "author" in data:
                if isinstance(data["author"], str):
                    metadata["author"] = data["author"]
                elif isinstance(data["author"], dict):
                    author_name = data["author"].get("name", "")
                    author_email = data["author"].get("email", "")
                    if author_name and author_email:
                        metadata["author"] = f"{author_name} <{author_email}>"
                    else:
                        metadata["author"] = author_name
            
            # Repository information
            if "repository" in data:
                if isinstance(data["repository"], str):
                    metadata["repository_url"] = data["repository"]
                elif isinstance(data["repository"], dict) and "url" in data["repository"]:
                    metadata["repository_url"] = data["repository"]["url"]
            
            # License
            metadata["license"] = data.get("license")
            
            # Homepage
            metadata["homepage"] = data.get("homepage")
            
            # Keywords for badges and section tags
            if "keywords" in data and isinstance(data["keywords"], list):
                metadata["keywords"] = data["keywords"]
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse package.json")

    # Extract information from existing README.md if available
    for path in repo_contents:
        if path.lower().endswith('readme.md'):
            logger.info("Analyzing existing README.md for metadata...")
            content = repo_contents[path]
            
            # Extract project name from first heading if not already found
            if not metadata["project_name"]:
                title_match = re.search(r'^#\s+(.+?)(\n|$)', content)
                if title_match:
                    metadata["project_name"] = title_match.group(1).strip()
            
            # Extract description from text after first heading if not already found
            if not metadata["description"]:
                desc_match = re.search(r'^#.*?\n+([^\n#]+)', content, re.MULTILINE)
                if desc_match:
                    metadata["description"] = desc_match.group(1).strip()
            
            # Look for badges in existing README
            badge_matches = re.findall(r'!\[(.*?)\]\((.*?)\)', content)
            for badge_text, badge_url in badge_matches:
                if "badge" in badge_url.lower() or "shield" in badge_url.lower():
                    metadata["badges"].append({"text": badge_text, "url": badge_url})
            
            break  # Only process the first README found

    # Clean up URLs if present
    for url_field in ["repository_url", "homepage"]:
        if metadata.get(url_field):
            # Remove git+ prefix and .git suffix
            cleaned_url = re.sub(r'^git\+', '', metadata[url_field])
            cleaned_url = re.sub(r'\.git$', '', cleaned_url)
            metadata[url_field] = cleaned_url

    logger.info(f"Metadata analysis complete. Fields found: {', '.join([k for k, v in metadata.items() if v])}")
    return metadata


def get_ai_readme_completion(prompt: str, api_key: str, config: dict) -> str:
    """
    Generate AI completion using multiple AI providers based on configuration
    """
    ai_provider = config.get("ai_provider", "openrouter")
    model_name = config.get("ai_model", "meta-llama/llama-3.2-3b-instruct:free")
    temperature = float(config.get("ai_temperature", 0.7))
    max_tokens = int(config.get("ai_max_tokens", 4000))
    
    # Verify API key is present
    if not api_key:
        logger.error(f"No API key provided for {ai_provider}")
        raise RuntimeError(f"Missing API key for {ai_provider}. Please check your .env file.")
        
    logger.info(f"Sending request to {ai_provider}. Model: {model_name}, Temp: {temperature}, Max Tokens: {max_tokens}")
    
    if ai_provider == "openrouter" or api_key.startswith("sk-or-"):
        # OpenRouter API
        try:
            # Construct request payload
            request_json = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert technical writer generating comprehensive, professional README.md files for GitHub repositories."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add debug log
            logger.info(f"Sending request to OpenRouter with model: {model_name}")
            if api_key.startswith('sk-or-'):
                logger.info("Using OpenRouter API key (verified format)")
            else:
                logger.warning("API key may not be in correct OpenRouter format")
                
            # Make the request
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/readme-generator",
                    "X-Title": "README Generator"
                },
                json=request_json,
                timeout=120  # Add a longer timeout for large generations
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("choices") and result["choices"][0].get("message"):
                    logger.info("AI response received successfully from OpenRouter.")
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    logger.warning("AI response malformed or empty, will try fallback.")
                    raise RuntimeError("AI response malformed or empty.")
            else:
                error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                
                # Special handling for rate limits
                if response.status_code == 429:
                    logger.warning("Rate limit hit. Proceeding with partial results.")
                    # Return a message that will be included in the README but not trigger an error in the UI
                    return """# README Generated Successfully

Note: API rate limits were encountered during generation, but we've still created a README for you. 
Some sections may be less detailed than usual.

**Try these options:**
- Wait a minute and try regenerating specific sections
- Try a different AI model from the dropdown
- Continue using this README as-is

---

Your project README content will appear here once API rate limits reset.
"""
                
                # Try fallback to a different free model if this wasn't already an attempt
                if "llama-3" not in model_name.lower() and "free" not in model_name.lower():
                    logger.info("Attempting fallback to free model...")
                    fallback_config = config.copy()
                    fallback_config["ai_model"] = "meta-llama/llama-3.2-3b-instruct:free"
                    return get_ai_readme_completion(prompt, api_key, fallback_config)
                
                raise RuntimeError(error_msg)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while contacting OpenRouter: {e}")
            raise RuntimeError(f"Network error: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from OpenRouter: {e}", exc_info=True)
            raise RuntimeError(f"Invalid response format from OpenRouter: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during OpenRouter completion: {e}", exc_info=True)
            raise RuntimeError(f"Error: {str(e)}. Check your OpenRouter API key and network connection.") from e
    
    elif ai_provider == "openai":
        # OpenAI API
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert technical writer generating README.md files."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                logger.info("AI response received successfully from OpenAI.")
                return response.choices[0].message.content.strip()
            raise RuntimeError(f"AI response malformed or empty. Finish reason: {response.choices[0].finish_reason if response.choices else 'N/A'}")
        except Exception as e:
            logger.error(f"An OpenAI API error occurred: {e}")
            raise RuntimeError(f"OpenAI API error: {e}") from e
    
    elif ai_provider == "anthropic":
        # Anthropic Claude API
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": model_name,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [
                        {"role": "user", "content": f"You are an expert technical writer. {prompt}"}
                    ]
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("content") and len(result["content"]) > 0:
                    logger.info("AI response received successfully from Anthropic.")
                    return result["content"][0]["text"].strip()
                else:
                    raise RuntimeError("AI response malformed or empty.")
            else:
                error_msg = f"Anthropic API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while contacting Anthropic: {e}")
            raise RuntimeError(f"Network error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic completion: {e}", exc_info=True)
            raise RuntimeError("An unexpected error occurred while communicating with Anthropic.") from e
    
    elif ai_provider == "google":
        # Google AI API
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}",
                headers={
                    "Content-Type": "application/json"
                },
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": f"You are an expert technical writer generating comprehensive README.md files. {prompt}"}
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("candidates") and len(result["candidates"]) > 0:
                    content = result["candidates"][0].get("content", {})
                    if content.get("parts") and len(content["parts"]) > 0:
                        logger.info("AI response received successfully from Google AI.")
                        return content["parts"][0]["text"].strip()
                raise RuntimeError("AI response malformed or empty.")
            else:
                error_msg = f"Google AI API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while contacting Google AI: {e}")
            raise RuntimeError(f"Network error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during Google AI completion: {e}", exc_info=True)
            raise RuntimeError("An unexpected error occurred while communicating with Google AI.") from e
    
    else:
        # Fallback to OpenRouter if provider not recognized
        logger.warning(f"Unknown AI provider '{ai_provider}', falling back to OpenRouter")
        config_copy = config.copy()
        config_copy["ai_provider"] = "openrouter"
        return get_ai_readme_completion(prompt, api_key, config_copy)

def analyze_file_structure(repo_contents: dict) -> dict:
    """
    Analyzes the repository file structure to create a directory tree representation.
    """
    logger.info("Analyzing file structure...")
    
    # Create a tree representation
    tree = {}
    for path in repo_contents.keys():
        parts = path.split('/')
        current = tree
        for part in parts[:-1]:  # Process directories
            if part not in current:
                current[part] = {}
            current = current[part]
        # Add file as leaf node
        file_name = parts[-1] if parts else path
        current[file_name] = None
    
    # Identify important file categories
    structure_info = {
        "tree": tree,
        "source_files": [],
        "test_files": [],
        "documentation_files": [],
        "config_files": [],
        "asset_files": []
    }
    
    for path in repo_contents.keys():
        path_lower = path.lower()
        
        # Source code files
        if any(path.endswith(ext) for ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.go', '.rs']):
            if not any(folder in path_lower for folder in ['/test/', '/tests/']):
                structure_info["source_files"].append(path)
                
        # Test files
        elif any(folder in path_lower for folder in ['/test/', '/tests/']) or path_lower.startswith('test_'):
            structure_info["test_files"].append(path)
            
        # Documentation files
        elif any(path.endswith(ext) for ext in ['.md', '.rst', '.txt']) or '/docs/' in path_lower:
            structure_info["documentation_files"].append(path)
            
        # Configuration files
        elif any(path.endswith(ext) for ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.conf']) or path in ['.env.example', '.gitignore']:
            structure_info["config_files"].append(path)
            
        # Asset files
        elif any(path.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.svg', '.gif', '.ico', '.css']):
            structure_info["asset_files"].append(path)
    
    # Generate a formatted directory tree representation
    formatted_tree = []
    def format_tree(tree, prefix="", is_last=True, level=0, max_depth=4):
        if level > max_depth:
            return
            
        items = list(tree.items())
        if not items:
            return
            
        for i, (name, subtree) in enumerate(items):
            is_last_item = i == len(items) - 1
            
            # Skip hidden files/folders unless they're important
            if name.startswith('.') and name not in ['.github', '.env.example']:
                continue
                
            if level == 0:
                formatted_tree.append(name)
            else:
                marker = '└── ' if is_last_item else '├── '
                formatted_tree.append(f"{prefix}{marker}{name}")
                
            if subtree is not None:  # If it's a directory
                next_prefix = prefix + ('    ' if is_last_item else '│   ')
                format_tree(subtree, next_prefix, is_last_item, level + 1, max_depth)
    
    format_tree(tree)
    structure_info["formatted_tree"] = formatted_tree
    
    logger.info(f"File structure analysis complete. Found {len(structure_info['source_files'])} source files.")
    return structure_info

def analyze_repo(repo_contents: dict) -> dict:
    """
    Consolidates all analysis functions into a single callable function.
    """
    logger.info("Starting full repository analysis...")
    analysis_results = {
        "dependencies": analyze_dependencies(repo_contents),
        "entry_points": analyze_entry_points(repo_contents),
        "configuration": analyze_configuration(repo_contents),
        "docker_info": analyze_docker(repo_contents),
        "metadata": analyze_project_metadata(repo_contents),
        "file_structure": analyze_file_structure(repo_contents),
    }
    logger.info("Full repository analysis complete.")
    return analysis_results

def generate_readme_from_analysis(analysis: dict, repo_contents: dict, api_key: str, config: dict) -> str:
    """
    Generates a README using pre-analyzed data.
    """
    logger.info("Generating README from pre-analyzed data...")

    # 2. Build the "Executive Summary" part of the prompt from analysis
    summary_parts = []
    if analysis["metadata"].get("project_name"):
        summary_parts.append(f"*   **Project Name:** {analysis['metadata']['project_name']}")
    if analysis["metadata"].get("description"):
        summary_parts.append(f"*   **Core Purpose:** {analysis['metadata']['description']}")

    detected_tech = []
    if analysis["dependencies"].get("python"): detected_tech.append("Python")
    if analysis["dependencies"].get("javascript"): detected_tech.append("JavaScript")
    if analysis["docker_info"].get("base_image"): detected_tech.append("Docker")
    if detected_tech:
        summary_parts.append(f"*   **Primary Language(s)/Technologies:** {', '.join(detected_tech)}")

    if analysis["dependencies"].get("python"):
        summary_parts.append(f"*   **Python Dependencies:** {', '.join(analysis['dependencies']['python'][:10])}")
    if analysis["dependencies"].get("javascript"):
        summary_parts.append(f"*   **JavaScript Dependencies:** {', '.join(analysis['dependencies']['javascript'][:10])}")
        
    # Add file structure information
    if analysis.get("file_structure") and analysis["file_structure"].get("formatted_tree"):
        tree_str = "\n".join(analysis["file_structure"]["formatted_tree"][:20])  # Limit to first 20 entries
        summary_parts.append(f"*   **Project File Structure:**\n```\n{tree_str}\n```")

    # 3. Build the "Actionable Insights" part of the prompt from analysis
    insights_parts = []
    
    # Environment Variables
    if analysis["configuration"].get("required_env_vars"):
        insights_parts.append("*   **Required Environment Variables:** " + ", ".join(analysis['configuration']['required_env_vars']))
    
    # Run Commands
    run_commands = analysis["entry_points"].get("run_commands", [])
    if run_commands:
        insights_parts.append("*   **Identified Run Commands:**\n" + "\n".join([f"        *   `{cmd}`" for cmd in run_commands]))
    
    # Docker Information
    if analysis["docker_info"].get("exposed_ports"):
        insights_parts.append("*   **Exposed Ports:** " + ", ".join(analysis['docker_info']['exposed_ports']))
    
    # Main Files
    if analysis["entry_points"].get("main_files"):
        insights_parts.append("*   **Main Entry Point Files:**\n" + "\n".join([f"        *   `{file}`" for file in analysis["entry_points"]["main_files"]]))
    
    # Source Files (important for badges and language stats)
    if analysis.get("file_structure") and analysis["file_structure"].get("source_files"):
        source_extensions = {}
        for file in analysis["file_structure"]["source_files"]:
            ext = file.split('.')[-1] if '.' in file else 'unknown'
            source_extensions[ext] = source_extensions.get(ext, 0) + 1
        
        # Only include if we have meaningful data
        if source_extensions:
            extensions_list = [f"{ext} ({count})" for ext, count in sorted(source_extensions.items(), key=lambda x: x[1], reverse=True)]
            insights_parts.append("*   **Source Code Files:** " + ", ".join(extensions_list))
    
    # Config Files (important for installation instructions)
    if analysis.get("file_structure") and analysis["file_structure"].get("config_files"):
        config_files = [file for file in analysis["file_structure"]["config_files"]]
        if config_files:
            insights_parts.append("*   **Configuration Files:**\n" + "\n".join([f"        *   `{file}`" for file in config_files[:5]]))
    
    # Test Files (useful for contribution guidelines)
    if analysis.get("file_structure") and analysis["file_structure"].get("test_files"):
        test_count = len(analysis["file_structure"]["test_files"])
        if test_count > 0:
            insights_parts.append(f"*   **Tests:** {test_count} test files identified")
            
    # Documentation Files
    if analysis.get("file_structure") and analysis["file_structure"].get("documentation_files"):
        doc_files = [file for file in analysis["file_structure"]["documentation_files"]]
        if doc_files:
            insights_parts.append("*   **Documentation Files:**\n" + "\n".join([f"        *   `{file}`" for file in doc_files[:5]]))

    # 4. Select key file snippets for the prompt
    key_files = ["pyproject.toml", "package.json", "requirements.txt", "Dockerfile", ".env.example", "Procfile"]
    key_files.extend(analysis["entry_points"].get("main_files", []))

    prompt_file_contents = []
    for path in sorted(repo_contents.keys(), key=lambda x: x in key_files, reverse=True):
        if len(prompt_file_contents) >= config.get("max_prompt_files", 15):
            break
        content = repo_contents.get(path, "")
        if not isinstance(content, str): continue

        snippet = content[:config.get("max_content_snippet_length", 600)]
        if len(content) > len(snippet): snippet += "\n... [TRUNCATED]"
        prompt_file_contents.append(f"--- File: {path} ---\n{snippet}\n--- End File: {path} ---")

    # 5. Construct the final prompt
    # Save context for potential section regeneration
    global LATEST_FULL_PROMPT_CONTEXT_OVERVIEW
    LATEST_FULL_PROMPT_CONTEXT_OVERVIEW = {
        "summary_parts": summary_parts.copy() if isinstance(summary_parts, list) else [],
        "insights_parts": insights_parts.copy() if isinstance(insights_parts, list) else [],
        "file_contents": prompt_file_contents.copy() if isinstance(prompt_file_contents, list) else []
    }
    
    prompt = f"""
You are an expert technical writer and software architect. Your task is to generate a comprehensive, professional README.md file based on the detailed project analysis provided below. Your primary role is to synthesize the provided facts into a visually appealing, information-rich document with advanced Markdown features.

**PROJECT EXECUTIVE SUMMARY:**
{chr(10).join(summary_parts)}

**ACTIONABLE INSIGHTS FOR README SECTIONS:**
{chr(10).join(insights_parts)}

**ADVANCED README REQUIREMENTS:**
1. **Project Header** - Start with an eye-catching header that includes:
   - A large title with emoji relevant to the project type
   - Badges showing: Build status, Latest version, License, Language/framework versions
   - A short, compelling tagline or one-line description
   - If suitable, add a project logo or banner image (use placeholder URL instructions)

2. **Table of Contents** - Create a clickable navigation with all major sections

3. **File Structure** - Include a well-formatted directory tree showing the project's important files and folders (with brief annotations)

4. **Installation & Setup** - Create a comprehensive, step-by-step guide with:
   - Prerequisite software/tools with version requirements
   - System requirements when applicable
   - Installation commands in proper code blocks with syntax highlighting
   - Dependency management (e.g., `pip install -r requirements.txt`, `npm install`)
   - Environment variable setup instructions if needed (create and populate `.env` file)

5. **Usage Examples** - Include:
   - Basic usage examples in code blocks
   - Command-line instructions when applicable
   - Screenshots or placeholders for UI-based applications (with placeholder instructions)

6. **Features & Screenshots** - Format as:
   - A feature table with feature name, description, and status columns
   - Instructions for adding screenshots with placeholders

7. **API Documentation** - If the project has an API:
   - Endpoint tables with method, URL, parameters, and description columns
   - Example requests and responses in code blocks
   - Authentication details if applicable

8. **Deployment** - Include:
   - Environment-specific deployment guides (development, staging, production)
   - Cloud platform instructions if applicable

9. **Visuals & Formatting** - Enhance with:
   - Proper heading hierarchy (H1, H2, H3)
   - Tables for structured information
   - Code blocks with syntax highlighting
   - Collapsible sections for lengthy content
   - Emojis for visual emphasis on section headers
   - Horizontal dividers to separate major sections

10. **Community & Contribution** - Add:
    - Clear contribution guidelines
    - Code of conduct reference
    - Contributors section with placeholder for contributor images
    - Contact information or community channels

11. **License & Attribution** - Include proper license information and credits

**RAW FILE SNIPPETS FOR REFERENCE:**
{chr(10).join(prompt_file_contents)}

Remember to use proper Markdown syntax and formatting to create a professional, visually appealing README that effectively showcases the project. Keep the content well-structured and highly readable.
"""

    logger.info("Enhanced AI prompt constructed with executive summary and actionable insights.")
    return get_ai_readme_completion(prompt, api_key, config)


def generate_readme_for_repo(repo_url: str, openai_api_key: str, github_token: str) -> str:
    """
    The main logic function, refactored to be callable from a web app.
    It takes repo URL and keys, performs all steps, and returns the generated README.
    Raises exceptions on failure.
    """
    if not github_token:
        raise ValueError("GitHub token is required.")
    if not openai_api_key:
        raise ValueError("OpenAI API key is required.")

    config = load_config()
    logger.info(f"Starting README generation for {repo_url}")

    try:
        owner, repo_name = parse_github_url(repo_url)
        g = github.Github(github_token)
        repo = g.get_repo(f"{owner}/{repo_name}")

        logger.info(f"Fetching contents for {repo.full_name}...")
        repo_contents = get_repo_contents(repo, config)
        if not repo_contents:
            logger.warning("Repository might be empty or all files were excluded.")

        analysis_results = analyze_repo(repo_contents)

        logger.info("Generating README with AI...")
        ai_generated_readme = generate_readme_from_analysis(analysis_results, repo_contents, openai_api_key, config)

        logger.info("Successfully generated README content.")
        return ai_generated_readme

    except ValueError as e:
        logger.error(f"Validation or parsing error: {e}")
        raise
    except github.GithubException as e:
        logger.error(f"GitHub API error: {e}")
        raise RuntimeError(f"Failed to access the GitHub repository. Check the URL and token permissions.") from e
    except (openai.APIError, RuntimeError) as e:
        logger.error(f"AI generation or runtime error: {e}")
        raise
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        raise RuntimeError("An unexpected critical error occurred during README generation.") from e

# --- CLI Specific Functions ---
# These functions will only be used when the script is run directly.

def get_multiline_input(prompt_message):
    # This function is for the CLI interactive mode
    print(prompt_message + " (Blank line to finish, :cancel to abort)")
    lines = []
    while True:
        try:
            line = input()
            if line == ":cancel": return None
            if line == "" and lines: break
            lines.append(line)
        except EOFError: break
    return "\n".join(lines)

def interactive_review(sections: list[dict], api_key: str, config: dict) -> str | None:
    # This function is for the CLI interactive mode
    # ... (Its implementation remains the same, but it now takes `config`)
    return "\n\n".join([f"## {s['heading']}\n{s['content']}" for s in sections]) # Simplified for brevity

def parse_readme_into_sections(markdown_content: str) -> list[dict]:
    # This function is for the CLI interactive mode
    # ... (Its implementation remains the same)
    return [{"heading": "Full Content", "content": markdown_content}] # Simplified for brevity

def cli_main():
    """The main function for running the script from the command line."""
    parser = argparse.ArgumentParser(description="GitHub README Generator using AI (CLI mode)")
    # ... (argparse setup remains the same)
    parser.add_argument("repo_url", help="Full URL of the public GitHub repository")
    args = parser.parse_args()

    # Setup logging for CLI
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load keys from .env or args
    load_dotenv()
    gh_token = os.getenv("GITHUB_TOKEN")
    oa_key = os.getenv("OPENAI_API_KEY")

    try:
        # Call the main logic function
        generated_readme = generate_readme_for_repo(args.repo_url, oa_key, gh_token)

        # The CLI can still have an interactive review mode if desired
        print("\n--- Generated README ---")
        print(generated_readme)
        print("\n----------------------")

        # Here you could call the interactive_review or just save the file
        # For simplicity, we'll just print it.

    except (ValueError, RuntimeError, openai.APIError, github.GithubException) as e:
        logger.error(f"Operation failed: {e}")
    except Exception as e:
        logger.error(f"A critical unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    cli_main()
