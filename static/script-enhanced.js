// Enhanced Script for GitHub README Generator UI

document.addEventListener('DOMContentLoaded', () => {
    // DOM Element References
    const form = document.getElementById('readme-form');
    const generateBtn = document.getElementById('generate-btn');
    const repoUrlInput = document.getElementById('repo_url');
    const aiProviderSelect = document.getElementById('ai_provider');
    const aiModelSelect = document.getElementById('ai_model');
    const pasteUrlBtn = document.getElementById('paste-url');

    const progressSection = document.getElementById('progress-section');
    const progressSteps = document.querySelectorAll('.progress-step');
    const progressFill = document.getElementById('progress-fill');
    const analysisSection = document.getElementById('analysis-section');
    const analysisOutput = document.getElementById('analysis-output');
    const resultsSection = document.getElementById('results-section');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorMessageDiv = document.getElementById('error-message');
    const readmeOutput = document.getElementById('readme-output');
    const readmePreview = document.getElementById('readme-preview');
    const finalizeSection = document.getElementById('finalize-section');
    const copyBtn = document.getElementById('copy-btn');
    const downloadBtn = document.getElementById('download-btn');
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    const sectionButtons = document.querySelectorAll('.btn-outline-small[data-section]');
    const toastContainer = document.getElementById('toast-container');

    let analysisData = null;
    let repoContents = null;
    let finalReadmeContent = '';

    // AI Provider and Model Configuration
    const aiModels = {
        openrouter: [
            { value: 'meta-llama/llama-3.2-3b-instruct:free', text: 'Llama 3.2 3B (Free)', free: true },
            { value: 'meta-llama/llama-3.2-1b-instruct:free', text: 'Llama 3.2 1B (Free)', free: true },
            { value: 'google/gemma-2-9b-it:free', text: 'Gemma 2 9B (Free)', free: true },
            { value: 'microsoft/phi-3-mini-128k-instruct:free', text: 'Phi-3 Mini (Free)', free: true },
            { value: 'meta-llama/llama-3.1-8b-instruct:free', text: 'Llama 3.1 8B (Free)', free: true },
            { value: 'gpt-4o-mini', text: 'GPT-4o Mini (Paid)', free: false },
            { value: 'claude-3-haiku', text: 'Claude 3 Haiku (Paid)', free: false }
        ],
        openai: [
            { value: 'gpt-4o-mini', text: 'GPT-4o Mini', free: false },
            { value: 'gpt-4o', text: 'GPT-4o', free: false },
            { value: 'gpt-3.5-turbo', text: 'GPT-3.5 Turbo', free: false }
        ],
        anthropic: [
            { value: 'claude-3-haiku-20240307', text: 'Claude 3 Haiku', free: false },
            { value: 'claude-3-sonnet-20240229', text: 'Claude 3 Sonnet', free: false },
            { value: 'claude-3-opus-20240229', text: 'Claude 3 Opus', free: false }
        ],
        google: [
            { value: 'gemini-1.5-flash', text: 'Gemini 1.5 Flash', free: false },
            { value: 'gemini-1.5-pro', text: 'Gemini 1.5 Pro', free: false },
            { value: 'gemini-pro', text: 'Gemini Pro', free: false }
        ]
    };

    // Update model dropdown when provider changes
    function updateModelOptions() {
        const provider = aiProviderSelect.value;
        const models = aiModels[provider] || [];
        
        aiModelSelect.innerHTML = '';
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.value;
            option.textContent = model.text;
            if (model.free) {
                option.classList.add('free-model');
            }
            aiModelSelect.appendChild(option);
        });
    }

    // Initialize model dropdown
    updateModelOptions();
    aiProviderSelect.addEventListener('change', updateModelOptions);

    // Toast notification system
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        let icon = '';
        switch(type) {
            case 'success':
                icon = '<i class="fas fa-check-circle"></i>';
                break;
            case 'error':
                icon = '<i class="fas fa-exclamation-circle"></i>';
                break;
            case 'warning':
                icon = '<i class="fas fa-exclamation-triangle"></i>';
                break;
            default:
                icon = '<i class="fas fa-info-circle"></i>';
        }
        
        toast.innerHTML = `${icon} <span>${message}</span>`;
        toastContainer.appendChild(toast);
        
        // Remove the toast after animation completes
        setTimeout(() => {
            toast.remove();
        }, 3500);
    }
    
    // Paste URL functionality
    if (pasteUrlBtn) {
        pasteUrlBtn.addEventListener('click', async () => {
            try {
                const text = await navigator.clipboard.readText();
                if (text && text.includes('github.com')) {
                    repoUrlInput.value = text;
                    showToast('URL pasted successfully!', 'success');
                } else {
                    showToast('Clipboard does not contain a valid GitHub URL', 'warning');
                }
            } catch (err) {
                showToast('Unable to access clipboard', 'error');
                console.error('Clipboard error:', err);
            }
        });
    }

    // Tab switching functionality
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;
            
            // Update active tab
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // Update active content
            tabContents.forEach(content => {
                content.classList.remove('active');
            });
            
            const activeContent = targetTab === 'source' ? readmeOutput : readmePreview;
            activeContent.classList.add('active');
            
            // If switching to preview, render markdown
            if (targetTab === 'preview' && finalReadmeContent) {
                renderMarkdownPreview(finalReadmeContent);
            }
        });
    });

    // Simple Markdown to HTML renderer
    function renderMarkdownPreview(markdown) {
        // This is a simplified version, for a real app use a proper markdown library
        let html = markdown
            // Headers
            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            // Bold
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Links
            .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank">$1</a>')
            // Code blocks
            .replace(/```([^`]+)```/g, '<pre><code>$1</code></pre>')
            // Inline code
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Lists
            .replace(/^\- (.*$)/gm, '<li>$1</li>')
            // Line breaks
            .replace(/\n/g, '<br>');
        
        readmePreview.innerHTML = html;
    }

    // Toast notification system
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = document.createElement('i');
        switch(type) {
            case 'success': 
                icon.className = 'fas fa-check-circle';
                break;
            case 'error':
                icon.className = 'fas fa-exclamation-circle';
                break;
            case 'warning':
                icon.className = 'fas fa-exclamation-triangle';
                break;
            default:
                icon.className = 'fas fa-info-circle';
        }
        
        const text = document.createElement('span');
        text.textContent = message;
        
        toast.appendChild(icon);
        toast.appendChild(text);
        toastContainer.appendChild(toast);
        
        // Remove toast after 3 seconds
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => {
                toastContainer.removeChild(toast);
            }, 300);
        }, 3000);
    }

    // Update progress UI
    function updateProgress(step, message) {
        // Update steps
        progressSteps.forEach((stepEl, index) => {
            if (index < step) {
                stepEl.classList.add('completed');
                stepEl.classList.remove('active');
            } else if (index === step) {
                stepEl.classList.add('active');
                stepEl.classList.remove('completed');
            } else {
                stepEl.classList.remove('active', 'completed');
            }
        });
        
        // Update progress bar
        const progressPercentage = ((step + 1) / progressSteps.length) * 100;
        progressFill.style.width = `${progressPercentage}%`;
    }

    // Form submission handler
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Reset UI state
        hideAllSections();
        clearResults();
        
        // Get form values
        const repoUrl = repoUrlInput.value.trim();
        const aiProvider = aiProviderSelect.value;
        const aiModel = aiModelSelect.value;
        
        if (!repoUrl) {
            showToast('Please enter a GitHub repository URL', 'error');
            return;
        }
        
        // Show progress section
        progressSection.classList.remove('hidden');
        updateProgress(0, 'Analyzing repository...');
        
        try {
            // Step 1: Repository Analysis
            const analysisResult = await analyzeRepository(repoUrl);
            analysisData = analysisResult.analysis;
            repoContents = analysisResult.repo_contents;
            
            // Show analysis results
            analysisSection.classList.remove('hidden');
            displayAnalysisResults(analysisData);
            
            // Update progress
            updateProgress(1, 'Generating README...');
            
            // Step 2: Generate README
            const readmeResult = await generateReadme(analysisData, repoContents, aiProvider, aiModel);
            finalReadmeContent = readmeResult.readme;
            
            // Show results
            resultsSection.classList.remove('hidden');
            readmeOutput.textContent = finalReadmeContent;
            finalizeSection.classList.remove('hidden');
            
            // Update progress
            updateProgress(2, 'Complete!');
            showToast('README generated successfully!', 'success');
            
        } catch (error) {
            console.error('Error:', error);
            errorMessageDiv.textContent = `Error: ${error.message || 'An unexpected error occurred'}`;
            errorMessageDiv.classList.remove('hidden');
            showToast('Failed to generate README', 'error');
        }
    });

    // Analyze repository
    async function analyzeRepository(repoUrl) {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                repo_url: repoUrl
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to analyze repository');
        }
        
        return response.json();
    }

    // Generate README
    async function generateReadme(analysis, repoContents, aiProvider, aiModel) {
        const response = await fetch('/api/generate_full', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                analysis: analysis,
                repo_contents: repoContents,
                ai_provider: aiProvider,
                ai_model: aiModel
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to generate README');
        }
        
        return response.json();
    }

    // Display analysis results
    function displayAnalysisResults(analysis) {
        let htmlContent = `<div class="analysis-summary">`;
        
        // Display language stats
        htmlContent += `
            <div class="analysis-item">
                <h4><i class="fas fa-code"></i> Languages Detected</h4>
                <p>${Object.keys(analysis.language_stats || {}).join(', ') || 'None detected'}</p>
            </div>
        `;
        
        // Display dependencies
        htmlContent += `
            <div class="analysis-item">
                <h4><i class="fas fa-cubes"></i> Dependencies</h4>
                <p>${analysis.dependencies?.length > 0 ? analysis.dependencies.join(', ') : 'None detected'}</p>
            </div>
        `;
        
        // Display project type
        htmlContent += `
            <div class="analysis-item">
                <h4><i class="fas fa-project-diagram"></i> Project Type</h4>
                <p>${analysis.project_type || 'Unknown'}</p>
            </div>
        `;
        
        htmlContent += `</div>`;
        analysisOutput.innerHTML = htmlContent;
    }

    // Copy README to clipboard
    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(finalReadmeContent)
            .then(() => {
                showToast('README copied to clipboard!', 'success');
            })
            .catch(err => {
                showToast('Failed to copy README', 'error');
                console.error('Copy error:', err);
            });
    });

    // Download README
    downloadBtn.addEventListener('click', () => {
        const blob = new Blob([finalReadmeContent], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'README.md';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast('README.md downloaded!', 'success');
    });

    // Section regeneration buttons
    sectionButtons.forEach(button => {
        button.addEventListener('click', async () => {
            const sectionName = button.dataset.section;
            if (!analysisData || !finalReadmeContent) {
                showToast('No README generated yet', 'warning');
                return;
            }
            
            button.disabled = true;
            const originalText = button.innerHTML;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Regenerating...';
            
            try {
                // Extract section content
                const sectionPattern = new RegExp(`## .*${sectionName}.*\\n([\\s\\S]*?)(?=\\n## |$)`, 'i');
                const match = finalReadmeContent.match(sectionPattern);
                const sectionContent = match ? match[1].trim() : '';
                const sectionHeading = `${sectionName.charAt(0).toUpperCase()}${sectionName.slice(1)}`;
                
                // Call API to regenerate section
                const response = await fetch('/api/regenerate_section', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        analysis: analysisData,
                        section_heading: sectionHeading,
                        section_content: sectionContent,
                        ai_provider: aiProviderSelect.value,
                        ai_model: aiModelSelect.value
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to regenerate section');
                }
                
                const result = await response.json();
                
                // Replace section in README
                const newContent = result.content;
                const updatedReadme = finalReadmeContent.replace(
                    sectionPattern,
                    `## ${sectionHeading}\n${newContent}\n`
                );
                
                // Update UI
                finalReadmeContent = updatedReadme;
                readmeOutput.textContent = finalReadmeContent;
                if (document.querySelector('.tab[data-tab="preview"]').classList.contains('active')) {
                    renderMarkdownPreview(finalReadmeContent);
                }
                
                showToast(`${sectionHeading} section regenerated!`, 'success');
            } catch (error) {
                console.error('Error regenerating section:', error);
                showToast('Failed to regenerate section', 'error');
            } finally {
                button.disabled = false;
                button.innerHTML = originalText;
            }
        });
    });

    // Helper functions
    function hideAllSections() {
        progressSection.classList.add('hidden');
        analysisSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        finalizeSection.classList.add('hidden');
    }
    
    function clearResults() {
        analysisOutput.innerHTML = '';
        readmeOutput.textContent = '';
        readmePreview.innerHTML = '';
        errorMessageDiv.textContent = '';
        errorMessageDiv.classList.add('hidden');
        analysisData = null;
        repoContents = null;
        finalReadmeContent = '';
    }
});
