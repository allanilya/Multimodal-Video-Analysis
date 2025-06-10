class VideoAnalysisApp {
    constructor() {
        this.currentVideoId = null;
        this.currentVideoData = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.setupTabs();
    }

    bindEvents() {
        // Process video button
        document.getElementById('processBtn').addEventListener('click', () => {
            this.processVideo();
        });

        // Enter key on URL input
        document.getElementById('videoUrl').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.processVideo();
            }
        });

        // Chat functionality
        document.getElementById('sendChatBtn').addEventListener('click', () => {
            this.sendChatMessage();
        });

        document.getElementById('chatInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendChatMessage();
            }
        });

        // Search functionality
        document.getElementById('searchBtn').addEventListener('click', () => {
            this.performSearch();
        });

        document.getElementById('searchInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.performSearch();
            }
        });
    }

    setupTabs() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const tabName = btn.dataset.tab;
                this.switchTab(tabName);
            });
        });
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}Tab`).classList.add('active');
    }

    async processVideo() {
        const videoUrl = document.getElementById('videoUrl').value.trim();
        if (!videoUrl) {
            this.showError('Please enter a YouTube URL');
            return;
        }

        if (!this.isValidYouTubeUrl(videoUrl)) {
            this.showError('Please enter a valid YouTube URL');
            return;
        }

        this.showLoading(true);
        document.getElementById('processBtn').disabled = true;

        try {
            const response = await fetch('/api/process-video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: videoUrl })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.currentVideoId = data.video_id;
                this.currentVideoData = data.video_data;
                this.displayVideoAnalysis(data.video_data, data.sections);
                this.showSuccess('Video processed successfully!');
            } else {
                this.showError(data.message || 'Failed to process video');
            }
        } catch (error) {
            console.error('Error processing video:', error);
            this.showError('Network error. Please try again.');
        } finally {
            this.showLoading(false);
            document.getElementById('processBtn').disabled = false;
        }
    }

    displayVideoAnalysis(videoData, sections) {
        // Hide URL input and show analysis
        document.getElementById('urlInputSection').style.display = 'none';
        document.getElementById('videoAnalysis').style.display = 'block';

        // Update video info
        document.getElementById('videoTitle').textContent = videoData.title;
        document.getElementById('videoDescription').textContent = videoData.description || 'No description available';
        document.getElementById('videoDuration').textContent = `Duration: ${this.formatDuration(videoData.duration)}`;
        document.getElementById('videoLink').href = videoData.url;

        // Display sections
        this.displaySections(sections);
    }

    displaySections(sections) {
        const sectionsList = document.getElementById('sectionsList');
        sectionsList.innerHTML = '';

        if (!sections || sections.length === 0) {
            sectionsList.innerHTML = '<p>No sections available</p>';
            return;
        }

        sections.forEach((section, index) => {
            const sectionItem = document.createElement('div');
            sectionItem.className = 'section-item';
            sectionItem.innerHTML = `
                <div class="section-title">${section.title}</div>
                <div class="section-time">
                    <i class="far fa-clock"></i> 
                    ${this.formatTime(section.start_time)} - ${this.formatTime(section.end_time)}
                </div>
                <div class="section-summary">${section.summary}</div>
            `;

            // Add click handler to jump to timestamp
            sectionItem.addEventListener('click', () => {
                this.jumpToTimestamp(section.start_time);
            });

            sectionsList.appendChild(sectionItem);
        });
    }

    async sendChatMessage() {
        const chatInput = document.getElementById('chatInput');
        const message = chatInput.value.trim();

        if (!message || !this.currentVideoId) {
            return;
        }

        // Add user message to chat
        this.addChatMessage(message, 'user');
        chatInput.value = '';

        // Show typing indicator
        const typingId = this.showTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    video_id: this.currentVideoId,
                    message: message
                })
            });

            const data = await response.json();

            // Remove typing indicator
            this.removeTypingIndicator(typingId);

            if (data.status === 'success') {
                this.addChatMessage(data.response.answer, 'bot', data.response.citations);
            } else {
                this.addChatMessage('Sorry, I encountered an error processing your message.', 'bot');
            }
        } catch (error) {
            console.error('Error sending chat message:', error);
            this.removeTypingIndicator(typingId);
            this.addChatMessage('Network error. Please try again.', 'bot');
        }
    }

    addChatMessage(message, sender, citations = null) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}`;

        const icon = sender === 'user' ? 'fas fa-user' : 'fas fa-robot';
        
        let citationsHtml = '';
        if (citations && citations.length > 0) {
            citationsHtml = '<div class="message-citations">';
            citations.forEach(citation => {
                if (citation.type === 'text') {
                    citationsHtml += `
                        <div class="citation-item" onclick="app.jumpToTimestamp(${citation.start_time})">
                            <span class="citation-time">${app.formatTime(citation.start_time)}</span> - ${citation.title}
                        </div>
                    `;
                } else {
                    citationsHtml += `
                        <div class="citation-item" onclick="app.jumpToTimestamp(${citation.timestamp})">
                            <span class="citation-time">${app.formatTime(citation.timestamp)}</span> - Visual content
                        </div>
                    `;
                }
            });
            citationsHtml += '</div>';
        }

        messageDiv.innerHTML = `
            <i class="${icon}"></i>
            <div class="message-content">
                <p>${message}</p>
                ${citationsHtml}
            </div>
        `;

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    showTypingIndicator() {
        const chatMessages = document.getElementById('chatMessages');
        const typingDiv = document.createElement('div');
        const typingId = 'typing-' + Date.now();
        typingDiv.id = typingId;
        typingDiv.className = 'chat-message bot';
        typingDiv.innerHTML = `
            <i class="fas fa-robot"></i>
            <div class="message-content">
                <p><i class="fas fa-ellipsis-h"></i> Thinking...</p>
            </div>
        `;

        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return typingId;
    }

    removeTypingIndicator(typingId) {
        const typingElement = document.getElementById(typingId);
        if (typingElement) {
            typingElement.remove();
        }
    }

    async performSearch() {
        const searchInput = document.getElementById('searchInput');
        const query = searchInput.value.trim();

        if (!query || !this.currentVideoId) {
            return;
        }

        const searchResults = document.getElementById('searchResults');
        searchResults.innerHTML = `
            <div class="loading-search">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Searching video content...</p>
            </div>
        `;

        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    video_id: this.currentVideoId,
                    query: query
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.displaySearchResults(data.results);
            } else {
                searchResults.innerHTML = `<div class="error-message">Search failed: ${data.message}</div>`;
            }
        } catch (error) {
            console.error('Error performing search:', error);
            searchResults.innerHTML = '<div class="error-message">Network error. Please try again.</div>';
        }
    }

    displaySearchResults(results) {
        const searchResults = document.getElementById('searchResults');

        if (!results || results.length === 0) {
            searchResults.innerHTML = '<p>No results found.</p>';
            return;
        }

        searchResults.innerHTML = '';

        results.forEach(result => {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'search-result-item';

            let content, timestamp;
            if (result.type === 'text') {
                content = result.title || 'Text content';
                timestamp = this.formatTime(result.start_time);
                resultDiv.addEventListener('click', () => {
                    this.jumpToTimestamp(result.start_time);
                });
            } else {
                content = result.description || 'Visual content';
                timestamp = this.formatTime(result.timestamp);
                resultDiv.addEventListener('click', () => {
                    this.jumpToTimestamp(result.timestamp);
                });
            }

            resultDiv.innerHTML = `
                <div class="result-type">${result.type} content</div>
                <div class="result-content">${content}</div>
                <div class="result-meta">
                    <span><i class="far fa-clock"></i> ${timestamp}</span>
                    <span class="result-score">${Math.round(result.score * 100)}% match</span>
                </div>
            `;

            searchResults.appendChild(resultDiv);
        });
    }

    jumpToTimestamp(timestamp) {
        if (this.currentVideoData) {
            const youtubeUrl = `${this.currentVideoData.url}&t=${Math.floor(timestamp)}s`;
            window.open(youtubeUrl, '_blank');
        }
    }

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const remainingSeconds = seconds % 60;

        if (hours > 0) {
            return `${hours}:${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
        } else {
            return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
        }
    }

    isValidYouTubeUrl(url) {
        const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com\/(watch\?v=|embed\/)|youtu\.be\/)[\w-]+/;
        return youtubeRegex.test(url);
    }

    showLoading(show) {
        const loadingIndicator = document.getElementById('loadingIndicator');
        loadingIndicator.style.display = show ? 'block' : 'none';
    }

    showError(message) {
        // Remove existing messages
        this.removeMessages();
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${message}`;
        
        document.querySelector('.url-input-section').appendChild(errorDiv);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 5000);
    }

    showSuccess(message) {
        // Remove existing messages
        this.removeMessages();
        
        const successDiv = document.createElement('div');
        successDiv.className = 'success-message';
        successDiv.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
        
        document.querySelector('.url-input-section').appendChild(successDiv);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            if (successDiv.parentNode) {
                successDiv.remove();
            }
        }, 3000);
    }

    removeMessages() {
        const messages = document.querySelectorAll('.error-message, .success-message');
        messages.forEach(msg => msg.remove());
    }
}

// Initialize the app when the page loads
const app = new VideoAnalysisApp();