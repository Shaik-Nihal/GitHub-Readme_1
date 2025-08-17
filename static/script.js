document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Document loaded - Initializing README Generator...');
    
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
    
    // Debug logging of key elements
    console.log('📊 DOM Element status:');
    console.log(' - Enhancement section:', enhancementSection ? 'Found ✅' : 'Not found ❌');
    console.log(' - Section cards:', document.querySelectorAll('.section-card').length);
    console.log(' - Copy button:', copyBtn ? 'Found ✅' : 'Not found ❌');
    console.log(' - Download button:', downloadBtn ? 'Found ✅' : 'Not found ❌');

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
            
            console.log('✅ README generation completed successfully');

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
        // Store the final README content globally
        finalReadmeContent = markdown;
        console.log('📄 README content stored:', finalReadmeContent ? 'Success' : 'Failed');
        
        // Convert basic markdown to HTML for preview
        let html = markdown
            .replace(/^#### (.*$)/gim, '<h4>$1</h4>')
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/gim, '<em>$1</em>')
            .replace(/```([\s\S]*?)```/gim, '<pre><code>$1</code></pre>')
            .replace(/`([^`]*?)`/gim, '<code>$1</code>')
            .replace(/\[([^\]]+)\]\(([^)]+)\)/gim, '<a href="$2" target="_blank">$1</a>')
            .replace(/^\* (.+$)/gim, '<li>$1</li>')
            .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
            .replace(/^\d+\. (.+$)/gim, '<li>$1</li>')
            .replace(/^> (.+$)/gim, '<blockquote>$1</blockquote>')
            .replace(/\n/gim, '<br>');

        // Clean up nested lists
        html = html.replace(/<\/li><br><li>/gim, '</li><li>');
        html = html.replace(/<ul><br>/gim, '<ul>');
        html = html.replace(/<br><\/ul>/gim, '</ul>');

        // Populate the markdown source tab
        const readmeOutput = document.getElementById('readme-output');
        if (readmeOutput) {
            readmeOutput.textContent = markdown;
            console.log('✅ Markdown source populated');
        }

        // Populate the HTML preview tab
        const readmePreview = document.getElementById('readme-preview');
        if (readmePreview) {
            readmePreview.innerHTML = html;
            console.log('✅ HTML preview populated');
        }

        // Show the copy and download buttons
        if (readmeHeaderActions) {
            readmeHeaderActions.classList.remove('hidden');
            console.log('✅ Copy/Download buttons shown');
        }

        // Initialize tab functionality
        initializeTabs();
        
        // Show the enhancement section with more robust handling
        console.log('🔍 Attempting to show enhancement section...');
        const enhancementSection = document.getElementById('enhancement-section');
        
        if (enhancementSection) {
            console.log('Enhancement section element found, showing it...');
            enhancementSection.classList.remove('hidden');
            enhancementSection.style.display = 'block';
            enhancementSection.style.visibility = 'visible';
            enhancementSection.style.opacity = '1';
            
            // Initialize regeneration functionality
            initializeSectionRegeneration();
            
            // Scroll to section after a delay
            setTimeout(() => {
                enhancementSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                console.log('✅ Scrolled to enhancement section');
            }, 1000);
            
            console.log('✅ Enhancement section shown and initialized');
        } else {
            console.error('❌ Enhancement section element not found in renderReadme');
            // Try to add a fallback notification
            if (typeof showToast === 'function') {
                showToast('Error: Enhancement section not found. Try using the "Force show section" button.', 'error');
            }
        }
    }

    // Function to show the enhancement section
    function showEnhancementSection() {
        console.log('🎯 Showing enhancement section...');
        
        if (enhancementSection) {
            enhancementSection.classList.remove('hidden');
            enhancementSection.style.display = 'block';
            enhancementSection.style.visibility = 'visible';
            enhancementSection.style.opacity = '1';
            
            // Initialize regeneration functionality
            initializeSectionRegeneration();
            
            // Scroll to section after a delay
            setTimeout(() => {
                enhancementSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 1000);
            
            console.log('✅ Enhancement section shown and initialized');
            return true;
        } else {
            console.error('❌ Enhancement section element not found');
            return false;
        }
    }

    // Initialize tab switching functionality
    function initializeTabs() {
        const tabs = document.querySelectorAll('.readme-tabs .tab');
        const tabContents = document.querySelectorAll('.tab-content');

        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs and contents
                tabs.forEach(t => t.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));

                // Add active class to clicked tab
                tab.classList.add('active');

                // Show corresponding content
                const tabType = tab.getAttribute('data-tab');
                const targetContent = document.getElementById(`readme-${tabType}`);
                if (targetContent) {
                    targetContent.classList.add('active');
                }
            });
        });
    }

    // Function to ensure regeneration section is properly shown
    // Making it globally accessible
    window.forceShowRegenerationSection = function() {
        console.log('🔄 Force showing regeneration section...');
        const section = document.getElementById('enhancement-section');
        
        if (section) {
            // Remove hidden class and force display
            section.classList.remove('hidden');
            section.style.display = 'block';
            section.style.visibility = 'visible';
            section.style.opacity = '1';
            
            // Initialize the regeneration functionality
            initializeSectionRegeneration();
            
            // Also show toast for user feedback
            if (typeof showToast === 'function') {
                showToast('Regeneration section visibility forced!', 'success');
            }
            
            // Scroll to the section
            setTimeout(() => {
                section.scrollIntoView({ behavior: 'smooth', block: 'start' });
                console.log('Scrolled to enhancement section');
            }, 500);
            
            console.log('✅ Force show completed successfully');
            return true;
        }
        
        console.error('Could not find regeneration section element');
        alert('Error: Could not find regeneration section element!');
        return false;
    }

    // Section regeneration functionality
    
    // Copy button functionality
    if (copyBtn) {
        copyBtn.addEventListener('click', async () => {
            console.log('📋 Copy button clicked');
            
            if (!finalReadmeContent) {
                console.error('No README content to copy');
                showToast('No README content available to copy', 'error');
                return;
            }
            
            try {
                await navigator.clipboard.writeText(finalReadmeContent);
                copyBtn.innerHTML = '<i class="fas fa-check"></i><span>Copied!</span>';
                showToast('README copied to clipboard!', 'success');
                setTimeout(() => {
                    copyBtn.innerHTML = '<i class="fas fa-copy"></i><span>Copy</span>';
                }, 2000);
                console.log('✅ README copied to clipboard');
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
                    copyBtn.innerHTML = '<i class="fas fa-check"></i><span>Copied!</span>';
                    showToast('README copied to clipboard!', 'success');
                    setTimeout(() => {
                        copyBtn.innerHTML = '<i class="fas fa-copy"></i><span>Copy</span>';
                    }, 2000);
                    console.log('✅ README copied using fallback method');
                } catch (err) {
                    console.error('Fallback copy failed:', err);
                    showToast('Failed to copy README', 'error');
                }
                document.body.removeChild(textArea);
            }
        });
        console.log('✅ Copy button event listener attached');
    } else {
        console.error('❌ Copy button not found');
    }

    // Download button functionality
    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            console.log('💾 Download button clicked');
            
            if (!finalReadmeContent) {
                console.error('No README content to download');
                showToast('No README content available to download', 'error');
                return;
            }
            
            try {
                const blob = new Blob([finalReadmeContent], { type: 'text/markdown' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'README.md';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                showToast('README.md downloaded successfully!', 'success');
                console.log('✅ README downloaded');
            } catch (err) {
                console.error('Failed to download README:', err);
                showToast('Failed to download README', 'error');
            }
        });
        console.log('✅ Download button event listener attached');
    } else {
        console.error('❌ Download button not found');
    }

    // Section regeneration functionality
    function initializeSectionRegeneration() {
        console.log('🔧 Initializing section regeneration...');
        const sectionCards = document.querySelectorAll('.section-card');
        console.log(`Found ${sectionCards.length} section cards`);
        
        sectionCards.forEach((card, index) => {
            const sectionType = card.getAttribute('data-section');
            const regenerateBtn = card.querySelector('.regenerate-btn');
            
            console.log(`Card ${index + 1}: ${sectionType}, Button:`, regenerateBtn);
            
            if (regenerateBtn) {
                // Remove any existing event listeners to prevent duplicates
                regenerateBtn.removeEventListener('click', regenerateBtn._regenerationHandler);
                
                // Create a named function reference for the handler
                regenerateBtn._regenerationHandler = async (e) => {
                    console.log(`🔄 Regenerate clicked for: ${sectionType}`);
                    e.stopPropagation();
                    await regenerateSection(card);
                };
                
                regenerateBtn.addEventListener('click', regenerateBtn._regenerationHandler);
                console.log(`✅ Event listener added for ${sectionType}`);
                
                // Add visual feedback
                regenerateBtn.style.cursor = 'pointer';
                regenerateBtn.title = `Regenerate ${sectionType} section`;
            } else {
                console.warn(`❌ No regenerate button found for ${sectionType}`);
            }
        });
        
        console.log('✅ Section regeneration initialization complete');
    }

    // Debug function to test regeneration (for development)
    window.testRegeneration = function() {
        console.log('🧪 testRegeneration function called');
        
        // Set some dummy content for testing
        if (!finalReadmeContent) {
            finalReadmeContent = `# Test README

## Description
This is a test README generated for testing the regeneration functionality.

## Features
- Feature 1
- Feature 2
- Feature 3

## Installation
\`\`\`bash
npm install
\`\`\`

## Usage
\`\`\`javascript
console.log('Hello World');
\`\`\`

## Contributing
Please contribute to this project.

## License
MIT License`;
            console.log('📝 Test README content set');
        }
        
        const enhancementSection = document.getElementById('enhancement-section');
        console.log('Enhancement section found:', enhancementSection);
        
        if (enhancementSection) {
            console.log('Before - Classes:', enhancementSection.classList.toString());
            console.log('Before - Display:', window.getComputedStyle(enhancementSection).display);
            
            enhancementSection.classList.remove('hidden');
            enhancementSection.style.display = 'block';
            enhancementSection.style.visibility = 'visible';
            enhancementSection.style.opacity = '1';
            
            console.log('After - Classes:', enhancementSection.classList.toString());
            console.log('After - Display:', window.getComputedStyle(enhancementSection).display);
            
            // Initialize regeneration
            initializeSectionRegeneration();
            
            // Show copy/download buttons
            if (readmeHeaderActions) {
                readmeHeaderActions.classList.remove('hidden');
                console.log('✅ Copy/Download buttons shown');
            }
            
            // Show toast
            if (typeof showToast === 'function') {
                showToast('Test mode activated! Regeneration section and buttons are ready.', 'success');
            } else {
                console.warn('showToast function not available');
                alert('Test mode activated! Regeneration section and buttons are ready.');
            }
            
            // Also render dummy content in the readme output for better testing
            const readmeOutput = document.getElementById('readme-output');
            if (readmeOutput) {
                readmeOutput.textContent = finalReadmeContent;
                console.log('✅ Test content populated in readme output');
            }
            
            // Scroll to the section
            setTimeout(() => {
                enhancementSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                console.log('Scrolled to enhancement section');
            }, 500);
            
            console.log('✅ Test regeneration setup complete');
            console.log('🎯 Try clicking regenerate buttons or copy/download buttons!');
            return true;
        } else {
            console.error('❌ Enhancement section not found');
            alert('Error: Could not find regeneration section');
            return false;
        }
    };

    async function regenerateSection(card) {
        const sectionType = card.getAttribute('data-section');
        const regenerateBtn = card.querySelector('.regenerate-btn');
        const progressIndicator = card.querySelector('.progress-indicator');
        
        if (!sectionType) {
            console.error('No section type found for card:', card);
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

            console.log(`Regenerating ${sectionType} section...`);
            
            // Get AI provider and model selections
            const selectedProvider = aiProviderSelect?.value || 'openrouter';
            const selectedModel = aiModelSelect?.value || 'meta-llama/llama-3.2-3b-instruct:free';

            const requestBody = {
                section_heading: sectionType,
                section_content: currentSectionContent,
                ai_provider: selectedProvider,
                ai_model: selectedModel,
                analysis: analysisData // Include analysis data if available
            };

            const response = await fetch('/api/regenerate_section', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
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