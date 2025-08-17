/**
 * Tab Preview Fix - This script fixes issues with the README preview functionality
 * This will override any existing preview tab handling to ensure it works properly
 */

// Wait for the document to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('🔧 Tab Preview Fix loaded - Version 1.1');

    // Initialize after a short delay to ensure other scripts have loaded
    setTimeout(initializeTabFix, 500);

    function initializeTabFix() {
        // Get references to key elements
        const tabs = document.querySelectorAll('.readme-tabs .tab');
        const previewTab = document.querySelector('.tab[data-tab="preview"]');
        const sourceTab = document.querySelector('.tab[data-tab="source"]');
        const previewContent = document.getElementById('readme-preview');
        const sourceContent = document.getElementById('readme-output');

        console.log('🔍 Found elements:', {
            'tabs': tabs.length,
            'previewTab': !!previewTab,
            'sourceTab': !!sourceTab,
            'previewContent': !!previewContent,
            'sourceContent': !!sourceContent
        });

        // Make sure we have the necessary elements
        if (!previewTab || !sourceTab || !previewContent || !sourceContent) {
            console.warn('⚠️ Missing required elements for tab preview fix');
            return;
        }

        // Add click listeners to the tabs
        tabs.forEach(tab => {
            // Remove existing event listeners
            const newTab = tab.cloneNode(true);
            tab.parentNode.replaceChild(newTab, tab);

            newTab.addEventListener('click', function() {
                const tabType = newTab.getAttribute('data-tab');
                console.log(`📋 Tab clicked: ${tabType}`);

                // Update active tab styling
                tabs.forEach(t => t.classList.remove('active'));
                newTab.classList.add('active');

                // Handle tab content visibility
                if (tabType === 'source') {
                    // Source tab - show original markdown
                    sourceContent.style.display = 'block';
                    previewContent.style.display = 'none';

                    // Make sure source content is up to date
                    if (window.finalReadmeContent) {
                        sourceContent.textContent = window.finalReadmeContent;
                    }
                } else if (tabType === 'preview') {
                    // Preview tab - render HTML preview
                    sourceContent.style.display = 'none';
                    previewContent.style.display = 'block';

                    // Clear existing content
                    previewContent.innerHTML = '';

                    // Get content from global variable or source content
                    const markdown = window.finalReadmeContent || sourceContent.textContent || '';
                    
                    if (markdown) {
                        // Convert to HTML
                        const html = convertMarkdown(markdown);
                        
                        // Create preview container
                        const previewContainer = document.createElement('div');
                        previewContainer.className = 'markdown-preview';
                        previewContainer.innerHTML = html;
                        
                        // Add to preview area
                        previewContent.appendChild(previewContainer);
                        
                        // Apply syntax highlighting
                        if (window.Prism) {
                            try {
                                window.Prism.highlightAllUnder(previewContent);
                            } catch (e) {
                                console.warn('⚠️ Error applying syntax highlighting:', e);
                            }
                        }
                    } else {
                        previewContent.innerHTML = '<div class="markdown-preview"><p>No content to preview</p></div>';
                    }
                }
            });
        });

        // Function to convert markdown to HTML
        function convertMarkdown(markdown) {
            // Use existing function if available
            if (typeof window.convertMarkdownToHTML === 'function') {
                return window.convertMarkdownToHTML(markdown);
            }
            
            // Fallback conversion function
            return markdown
                // Escape HTML
                .replace(/</g, '&lt;').replace(/>/g, '&gt;')
                
                // Headers
                .replace(/^##### (.*$)/gim, '<h5>$1</h5>')
                .replace(/^#### (.*$)/gim, '<h4>$1</h4>')
                .replace(/^### (.*$)/gim, '<h3>$1</h3>')
                .replace(/^## (.*$)/gim, '<h2>$1</h2>')
                .replace(/^# (.*$)/gim, '<h1>$1</h1>')
                
                // Bold and italic
                .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/gim, '<em>$1</em>')
                
                // Code blocks
                .replace(/```([a-zA-Z]*)\n([\s\S]*?)```/gm, function(match, language, code) {
                    return `<pre><code class="language-${language || 'plaintext'}">${code.trim()}</code></pre>`;
                })
                
                // Inline code
                .replace(/`([^`]*?)`/gim, '<code>$1</code>')
                
                // Lists - basic handling
                .replace(/^\* (.+)$/gim, '<li>$1</li>')
                .replace(/^- (.+)$/gim, '<li>$1</li>')
                .replace(/^(\d+)\. (.+)$/gim, '<li>$2</li>')
                
                // Blockquotes
                .replace(/^> (.+)$/gim, '<blockquote>$1</blockquote>')
                
                // Links
                .replace(/\[([^\]]+)\]\(([^)]+)\)/gim, '<a href="$2" target="_blank">$1</a>')
                
                // Paragraphs - simple approach
                .replace(/\n\n/gim, '</p><p>')
                .replace(/\n/gim, '<br>');
        }

        console.log('✅ Tab preview fix initialized successfully');
    }
});
