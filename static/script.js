document.addEventListener('DOMContentLoaded', () => {
    // DOM Element References
    const form = document.getElementById('readme-form');
    const generateBtn = document.getElementById('generate-btn');
    const repoUrlInput = document.getElementById('repo_url');
    const aiProviderSelect = document.getElementById('ai_provider');
    const aiModelSelect = document.getElementById('ai_model');

    const progressSection = document.getElementById('progress-section');
    const progressSteps = document.querySelectorAll('.progress-step');
    const analysisSection = document.getElementById('analysis-section');
    const analysisOutput = document.getElementById('analysis-output');
    const resultsSection = document.getElementById('results-section');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorMessageDiv = document.getElementById('error-message');
    const readmeOutput = document.getElementById('readme-output');
    const finalizeSection = document.getElementById('finalize-section');
    const enhancementSection = document.getElementById('enhancement-section');
    const copyBtn = document.getElementById('copy-btn');
    const downloadBtn = document.getElementById('download-btn');
    const readmeHeaderActions = document.getElementById('readme-header-actions');
    const testRegenerationBtn = document.getElementById('test-regeneration-btn');

    let analysisData = null;
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
                option.style.fontWeight = 'bold';
                option.style.color = '#28a745';
            }
            aiModelSelect.appendChild(option);
        });
        
        // Select first model by default
        if (models.length > 0) {
            aiModelSelect.value = models[0].value;
        }
    }

    // Initialize model options on page load
    if (aiProviderSelect && aiModelSelect) {
        updateModelOptions();
        aiProviderSelect.addEventListener('change', updateModelOptions);
    }

    // Progress Management
    function updateProgress(step) {
        progressSteps.forEach((el, index) => {
            el.classList.toggle('active', index <= step);
            el.classList.toggle('completed', index < step);
        });
    }

    // Form Submission Handler
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing Repository...';
        
        // Reset UI
        analysisSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        finalizeSection.classList.add('hidden');
        enhancementSection.classList.add('hidden');
        readmeHeaderActions.classList.add('hidden');
        errorMessageDiv.classList.add('hidden');
        progressSection.classList.remove('hidden');
        updateProgress(0);

        // Prepare data for API calls
        const initialData = {
            repo_url: repoUrlInput.value
        };

        try {
            // Step 1: Analyze Repository
            const analyzeResponse = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(initialData),
            });
            const analyzeResult = await analyzeResponse.json();
            if (!analyzeResponse.ok) throw new Error(analyzeResult.error || 'Failed to analyze repository.');

            analysisData = analyzeResult.analysis;
            displayAnalysis(analysisData);
            analysisSection.classList.remove('hidden');
            updateProgress(1);
            
            generateBtn.innerHTML = '<i class="fas fa-robot fa-spin"></i> Generating README...';
            loadingSpinner.classList.remove('hidden');
            resultsSection.classList.remove('hidden');

            // Step 2: Generate README
            const generateData = {
                analysis: analysisData,
                repo_contents: analyzeResult.repo_contents,
                ai_provider: aiProviderSelect.value,
                ai_model: aiModelSelect.value
            };

            const generateResponse = await fetch('/api/generate_full', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(generateData),
            });
            const generateResult = await generateResponse.json();
            if (!generateResponse.ok) throw new Error(generateResult.error || 'Failed to generate README.');

            finalReadmeContent = generateResult.readme;
            renderReadme(finalReadmeContent);
            updateProgress(2);
            
            // DEBUG: Show regeneration section with detailed logging
            console.log('README generation completed, showing enhancement section...');
            console.log('Enhancement section element:', enhancementSection);
            console.log('Enhancement section exists:', enhancementSection !== null);
            
            if (enhancementSection) {
                console.log('Before: classList =', enhancementSection.classList.toString());
                enhancementSection.classList.remove('hidden');
                console.log('After: classList =', enhancementSection.classList.toString());
                
                // Force visibility with multiple methods
                enhancementSection.style.display = 'block';
                enhancementSection.style.visibility = 'visible';
                enhancementSection.style.opacity = '1';
                
                console.log('Enhancement section should now be visible');
                console.log('Computed style display:', window.getComputedStyle(enhancementSection).display);
                console.log('Computed style visibility:', window.getComputedStyle(enhancementSection).visibility);
                
                // Scroll to section after a short delay
                setTimeout(() => {
                    enhancementSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 500);
            } else {
                console.error('Enhancement section element not found!');
            }
            
            if (readmeHeaderActions) {
                readmeHeaderActions.classList.remove('hidden');
                console.log('Header actions shown');
            } else {
                console.error('Header actions element not found!');
            }

        } catch (error) {
            console.error('Error during generation process:', error);
            errorMessageDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> Error: ${error.message}`;
            errorMessageDiv.classList.remove('hidden');
            resultsSection.classList.remove('hidden');
            progressSection.classList.add('hidden');
        } finally {
            loadingSpinner.classList.add('hidden');
            generateBtn.disabled = false;
            generateBtn.innerHTML = '<i class="fas fa-robot"></i> Generate README';
        }
    });

    function displayAnalysis(analysis) {
        let html = '<div class="analysis-summary">';
        
        if (analysis.dependencies) {
            html += '<div class="analysis-item">';
            html += '<h4><i class="fas fa-puzzle-piece"></i> Dependencies Found</h4>';
            if (analysis.dependencies.python && analysis.dependencies.python.length > 0) {
                html += `<p><strong>Python:</strong> ${analysis.dependencies.python.slice(0, 5).join(', ')}${analysis.dependencies.python.length > 5 ? '...' : ''}</p>`;
            }
            if (analysis.dependencies.javascript && analysis.dependencies.javascript.length > 0) {
                html += `<p><strong>JavaScript:</strong> ${analysis.dependencies.javascript.slice(0, 5).join(', ')}${analysis.dependencies.javascript.length > 5 ? '...' : ''}</p>`;
            }
            html += '</div>';
        }

        if (analysis.entry_points && analysis.entry_points.run_commands && analysis.entry_points.run_commands.length > 0) {
            html += '<div class="analysis-item">';
            html += '<h4><i class="fas fa-play"></i> Run Commands</h4>';
            html += `<p>${analysis.entry_points.run_commands.slice(0, 3).join(', ')}</p>`;
            html += '</div>';
        }

        if (analysis.metadata) {
            html += '<div class="analysis-item">';
            html += '<h4><i class="fas fa-info-circle"></i> Project Info</h4>';
            if (analysis.metadata.project_type) {
                html += `<p><strong>Type:</strong> ${analysis.metadata.project_type}</p>`;
            }
            if (analysis.metadata.frameworks && analysis.metadata.frameworks.length > 0) {
                html += `<p><strong>Frameworks:</strong> ${analysis.metadata.frameworks.join(', ')}</p>`;
            }
            html += '</div>';
        }

        html += '</div>';
        analysisOutput.innerHTML = html;
    }

    function renderReadme(markdown) {
        // Convert basic markdown to HTML for display
        let html = markdown
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            .replace(/\*\*(.*)\*\*/gim, '<strong>$1</strong>')
            .replace(/\*(.*)\*/gim, '<em>$1</em>')
            .replace(/```([\\s\\S]*?)```/gim, '<pre><code>$1</code></pre>')
            .replace(/`([^`]*)`/gim, '<code>$1</code>')
            .replace(/\\n/gim, '<br>');

        readmeOutput.innerHTML = `
            <div class="readme-preview">
                <div class="readme-toolbar">
                    <span class="readme-title"><i class="fab fa-markdown"></i> README.md</span>
                    <div class="readme-actions">
                        <button onclick="editSection(event)" class="edit-btn">
                            <i class="fas fa-edit"></i> Edit
                        </button>
                    </div>
                </div>
                <div class="readme-content">${html}</div>
            </div>
            <div class="readme-raw">
                <h4>Raw Markdown:</h4>
                <pre><code>${markdown}</code></pre>
            </div>
        `;
    }

    // Function to ensure regeneration section is properly shown
    function forceShowRegenerationSection() {
        console.log('Force showing regeneration section...');
        const section = document.getElementById('enhancement-section');
        if (section) {
            section.classList.remove('hidden');
            section.style.display = 'block';
            section.style.visibility = 'visible';
            section.style.opacity = '1';
            
            // Also initialize the regeneration functionality
            initializeSectionRegeneration();
            
            console.log('Regeneration section forced to show');
            return true;
        }
        console.error('Could not find regeneration section');
        return false;
    }

    // Test regeneration section button - Enhanced version
    if (testRegenerationBtn) {
        testRegenerationBtn.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('Test button clicked');
            
            const success = forceShowRegenerationSection();
            if (success) {
                testRegenerationBtn.style.display = 'none';
                alert('Regeneration section is now visible! Check below the form.');
            } else {
                alert('Could not show regeneration section - check console for errors');
            }
        });
    } else {
        console.error('Test regeneration button not found!');
    }

    // Copy button functionality
    if (copyBtn) {
        copyBtn.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(finalReadmeContent);
                copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                setTimeout(() => {
                    copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy Final README';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy text: ', err);
                // Fallback for older browsers
                const textArea = document.createElement("textarea");
                textArea.value = finalReadmeContent;
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                try {
                    document.execCommand('copy');
                    copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                    setTimeout(() => {
                        copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy Final README';
                    }, 2000);
                } catch (err) {
                    alert('Failed to copy text');
                }
                document.body.removeChild(textArea);
            }
        });
    }

    // Download button functionality
    if (downloadBtn) {
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
        });
    }

    // Section regeneration functionality
    function initializeSectionRegeneration() {
        const sectionCards = document.querySelectorAll('.section-card');
        
        sectionCards.forEach(card => {
            const regenerateBtn = card.querySelector('.regenerate-btn');
            if (regenerateBtn) {
                regenerateBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    await regenerateSection(card);
                });
            }
        });
    }

    async function regenerateSection(card) {
        const sectionType = card.getAttribute('data-section');
        const regenerateBtn = card.querySelector('.regenerate-btn');
        const progressIndicator = card.querySelector('.progress-indicator');
        
        if (!sectionType) {
            showToast('Error: Section type not found', 'error');
            return;
        }

        // Show loading state with animation
        card.classList.add('regenerating');
        regenerateBtn.innerHTML = '<i class="fas fa-sync-alt fa-spin"></i>';
        
        // Add glowing effect to indicate active regeneration
        card.style.boxShadow = '0 0 20px rgba(0, 217, 255, 0.6)';
        
        showToast(`Regenerating ${sectionType} section...`, 'info');
        
        try {
            // Extract current section content from the README
            let currentSectionContent = '';
            if (finalReadmeContent) {
                // Try to extract the current section content
                const sectionRegex = new RegExp(`##\\s*${sectionType}[\\s\\S]*?(?=##|$)`, 'i');
                const match = finalReadmeContent.match(sectionRegex);
                currentSectionContent = match ? match[0] : `## ${sectionType}\n\nCurrent content for ${sectionType} section.`;
            } else {
                currentSectionContent = `## ${sectionType}\n\nPlease provide content for the ${sectionType} section.`;
            }

            console.log(`Regenerating ${sectionType} section with content:`, currentSectionContent);

            const response = await fetch('/api/regenerate_section', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    section_heading: sectionType,
                    section_content: currentSectionContent,
                    ai_provider: aiProviderSelect?.value || 'openrouter',
                    ai_model: aiModelSelect?.value || 'meta-llama/llama-3.2-3b-instruct:free',
                    analysis: analysisData // Include analysis data if available
                })
            });

            const result = await response.json();
            
            if (!response.ok) {
                console.error('Regeneration API error:', result);
                throw new Error(result.error || 'Failed to regenerate section');
            }

            console.log('Regeneration result:', result);

            // Check if we have regenerated content
            if (result.content) {
                // Update the specific section in the README content
                const sectionRegex = new RegExp(`(##\\s*${sectionType}\\s*\\n)[\\s\\S]*?(?=\\n##|$)`, 'i');
                const newSectionContent = `## ${sectionType}\n\n${result.content.trim()}`;
                
                if (finalReadmeContent && sectionRegex.test(finalReadmeContent)) {
                    // Replace existing section
                    finalReadmeContent = finalReadmeContent.replace(sectionRegex, newSectionContent);
                } else if (finalReadmeContent) {
                    // Add new section at the end
                    finalReadmeContent += `\n\n${newSectionContent}`;
                } else {
                    // Create new README with just this section
                    finalReadmeContent = newSectionContent;
                }
                
                // Re-render the updated README
                renderReadme(finalReadmeContent);
                showToast(`${sectionType.charAt(0).toUpperCase() + sectionType.slice(1)} section updated successfully!`, 'success');
                
                // Show success animation
                card.classList.add('success-animation');
                setTimeout(() => {
                    card.classList.remove('success-animation');
                }, 600);
                
            } else if (result.warning) {
                showToast(result.warning, 'warning');
            } else {
                showToast('Section regenerated but content not returned', 'warning');
                console.warn('No content found in regeneration result:', result);
            }

        } catch (error) {
            console.error('Error regenerating section:', error);
            showToast(`Failed to regenerate ${sectionType}: ${error.message}`, 'error');
        } finally {
            // Remove regenerating state and restore normal appearance
            card.classList.remove('regenerating');
            card.style.boxShadow = '';
            regenerateBtn.innerHTML = '<i class="fas fa-sync-alt"></i>';
        }
    }

    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        `;
        
        const container = document.getElementById('toast-container') || document.body;
        container.appendChild(toast);
        
        // Show toast
        setTimeout(() => toast.classList.add('show'), 100);
        
        // Hide and remove toast
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => container.removeChild(toast), 300);
        }, 4000);
    }

    // Initialize section regeneration when enhancement section is shown
    const originalShowEnhancement = () => {
        enhancementSection.classList.remove('hidden');
        readmeHeaderActions.classList.remove('hidden');
        initializeSectionRegeneration();
    };

    // Override the section showing logic
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'attributes' && 
                mutation.attributeName === 'class' && 
                !enhancementSection.classList.contains('hidden')) {
                initializeSectionRegeneration();
            }
        });
    });
    
    if (enhancementSection) {
        observer.observe(enhancementSection, { attributes: true });
    }
});

// CSS for toast notifications - moved outside DOMContentLoaded
const toastStyles = `
    .toast-container {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
    }
    
    .toast {
        background: var(--dark-bg-secondary);
        border: 2px solid var(--border-color);
        border-radius: var(--radius-lg);
        color: var(--text-primary);
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        min-width: 300px;
        opacity: 0;
        transform: translateX(100%);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        box-shadow: var(--shadow-lg);
    }
    
    .toast.show {
        opacity: 1;
        transform: translateX(0);
    }
    
    .toast-success {
        border-color: #10b981;
        background: linear-gradient(135deg, var(--dark-bg-secondary), rgba(16, 185, 129, 0.1));
    }
    
    .toast-error {
        border-color: #ef4444;
        background: linear-gradient(135deg, var(--dark-bg-secondary), rgba(239, 68, 68, 0.1));
    }
    
    .toast-warning {
        border-color: #f59e0b;
        background: linear-gradient(135deg, var(--dark-bg-secondary), rgba(245, 158, 11, 0.1));
    }
    
    .toast i {
        font-size: 1.25rem;
    }
    
    .toast-success i {
        color: #10b981;
    }
    
    .toast-error i {
        color: #ef4444;
    }
    
    .toast-warning i {
        color: #f59e0b;
    }
`;

// Add toast styles to the page
if (!document.getElementById('toast-styles')) {
    const style = document.createElement('style');
    style.id = 'toast-styles';
    style.textContent = toastStyles;
    document.head.appendChild(style);
}

// Global function for section editing (if needed later)
function editSection(event) {
    // Placeholder for future section editing functionality
    alert('Section editing feature coming soon!');
}