document.addEventListener('DOMContentLoaded', () => {
    const messagesContainer = document.getElementById('messages');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const themeToggle = document.getElementById('themeToggle');
    const toolButtons = document.querySelectorAll('.tool-btn');
    const maxTokensInput = document.getElementById('maxTokens');
    const maxTokensValue = document.getElementById('maxTokensValue');
    const temperatureInput = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperatureValue');
    const modelSelect = document.getElementById('model');
    const saveSettingsButton = document.getElementById('saveSettings');

    // Configure marked options
    marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: false,
        mangle: false
    });

    // Theme handling
    const setTheme = (isDark) => {
        document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
        themeToggle.textContent = isDark ? '☀️' : '🌙';
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    };

    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme === 'dark');

    themeToggle.addEventListener('click', () => {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        setTheme(!isDark);
    });

    // Settings handling
    const updateSettings = () => {
        const settings = {
            maxTokens: parseInt(maxTokensInput.value),
            temperature: parseFloat(temperatureInput.value),
            model: modelSelect.value
        };
        localStorage.setItem('chatSettings', JSON.stringify(settings));
    };

    const loadSettings = () => {
        const savedSettings = JSON.parse(localStorage.getItem('chatSettings') || '{}');
        maxTokensInput.value = savedSettings.maxTokens || 1000;
        maxTokensValue.textContent = maxTokensInput.value;
        temperatureInput.value = savedSettings.temperature || 0.7;
        temperatureValue.textContent = temperatureInput.value;
        modelSelect.value = savedSettings.model || 'gpt-3.5-turbo';
    };

    maxTokensInput.addEventListener('input', () => {
        maxTokensValue.textContent = maxTokensInput.value;
    });

    temperatureInput.addEventListener('input', () => {
        temperatureValue.textContent = temperatureInput.value;
    });

    saveSettingsButton.addEventListener('click', updateSettings);

    loadSettings();

    // Tool button handling
    toolButtons.forEach(button => {
        button.addEventListener('click', () => {
            const command = button.dataset.command;
            messageInput.value = `/${command} `;
            messageInput.focus();
        });
    });

    // Message handling
    const addMessage = (content, isUser = false) => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
        
        if (isUser) {
            messageDiv.textContent = content;
        } else {
            messageDiv.innerHTML = marked.parse(content);
        }
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        return messageDiv;
    };

    const addLoadingIndicator = () => {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant-message loading';
        loadingDiv.textContent = '...';
        messagesContainer.appendChild(loadingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        return loadingDiv;
    };

    // File upload handling
    const fileInput = document.getElementById('fileInput');
    const uploadedFiles = document.getElementById('uploadedFiles');
    const uploadedDocuments = new Set();

    const addFileToList = (file) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-name">
                <i class="fas fa-file"></i>
                <span>${file.name}</span>
            </div>
            <div class="file-actions">
                <button class="remove-file" title="Remove file">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        fileItem.querySelector('.remove-file').addEventListener('click', () => {
            uploadedDocuments.delete(file.name);
            fileItem.remove();
        });

        uploadedFiles.appendChild(fileItem);
    };

    fileInput.addEventListener('change', async (e) => {
        const files = Array.from(e.target.files);
        
        for (const file of files) {
            if (uploadedDocuments.has(file.name)) continue;
            
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload_document', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Failed to upload file');
                }

                uploadedDocuments.add(file.name);
                addFileToList(file);
            } catch (error) {
                addMessage(`Error uploading ${file.name}: ${error.message}`);
            }
        }

        fileInput.value = '';
    });

    // Update message handling to include RAG context
    const sendMessage = async () => {
        const message = messageInput.value.trim();
        if (!message) return;

        addMessage(message, true);
        messageInput.value = '';
        const loadingIndicator = addLoadingIndicator();

        try {
            const settings = JSON.parse(localStorage.getItem('chatSettings') || '{}');
            const response = await fetch('/send_message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message,
                    settings,
                    useRag: message.startsWith('@'),
                    documents: Array.from(uploadedDocuments)
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Network response was not ok');
            }
            
            const data = await response.json();
            loadingIndicator.remove();
            addMessage(data.response);
        } catch (error) {
            loadingIndicator.remove();
            addMessage(`Error: ${error.message}`);
        }
    };

    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = messageInput.scrollHeight + 'px';
    });

    // Add welcome message
    addMessage("👋 Welcome! I'm your AI assistant. You can chat with me directly or use commands like /analyze, /translate, etc. for specific tasks.");
}); 