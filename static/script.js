class ChatApp {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.stats = {
            totalQueries: 0,
            weatherRequests: 0,
            aqiRequests: 0
        };

        this.initializeElements();
        this.connectWebSocket();
        this.bindEvents();
    }

    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }

    initializeElements() {
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatMessages = document.getElementById('chatMessages');
        this.connectionStatus = document.getElementById('connectionStatus');
        this.totalQueriesEl = document.getElementById('totalQueries');
        this.weatherRequestsEl = document.getElementById('weatherRequests');
        this.aqiRequestsEl = document.getElementById('aqiRequests');
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;

        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = () => {
            console.log('Connected to chat server');
            this.updateStatus('Connected', 'connected');
            this.reconnectAttempts = 0;
        };

        this.socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.addBotMessage(data.message);

                // Update stats if provided
                if (data.session_stats) {
                    this.updateStatsFromServer(data.session_stats);
                }
            } catch (error) {
                console.error('Error parsing message:', error);
            }
        };

        this.socket.onclose = () => {
            console.log('Disconnected from chat server');
            this.updateStatus('Disconnected', 'disconnected');
            this.attemptReconnect();
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('Connection Error', 'error');
        };
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            this.updateStatus(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`, 'reconnecting');

            setTimeout(() => {
                this.connectWebSocket();
            }, 2000 * this.reconnectAttempts);
        } else {
            this.updateStatus('Connection Failed', 'failed');
        }
    }

    bindEvents() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Enable/disable send button based on connection
        this.messageInput.addEventListener('input', () => {
            this.sendButton.disabled = !this.messageInput.value.trim() ||
                                     this.socket?.readyState !== WebSocket.OPEN;
        });
    }

    sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.socket?.readyState !== WebSocket.OPEN) return;

        this.addUserMessage(message);
        this.updateLocalStats(message);

        try {
            this.socket.send(JSON.stringify({
                message: message,
                timestamp: new Date().toISOString()
            }));
        } catch (error) {
            console.error('Error sending message:', error);
            this.addBotMessage('Sorry, there was an error sending your message. Please try again.');
        }

        this.messageInput.value = '';
        this.sendButton.disabled = true;
    }

    addUserMessage(message) {
        const messageEl = document.createElement('div');
        messageEl.className = 'user-message';
        messageEl.innerHTML = `
            <div class="message-avatar">👤</div>
            <div class="message-content">${this.escapeHtml(message)}</div>
        `;
        this.chatMessages.appendChild(messageEl);
        this.scrollToBottom();
    }

    addBotMessage(message) {
        const messageEl = document.createElement('div');
        messageEl.className = 'bot-message';
        messageEl.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content">${this.formatBotMessage(message)}</div>
        `;
        this.chatMessages.appendChild(messageEl);
        this.scrollToBottom();
    }

    formatBotMessage(message) {
        return this.escapeHtml(message)
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }

    updateLocalStats(message) {
        this.stats.totalQueries++;

        const lowerMessage = message.toLowerCase();
        if (lowerMessage.includes('weather')) {
            this.stats.weatherRequests++;
        }
        if (lowerMessage.includes('aqi') || lowerMessage.includes('air quality')) {
            this.stats.aqiRequests++;
        }

        this.updateStatsDisplay();
    }

    updateStatsFromServer(serverStats) {
        if (serverStats.message_count) {
            this.stats.totalQueries = serverStats.message_count;
            this.updateStatsDisplay();
        }
    }

    updateStatsDisplay() {
        this.totalQueriesEl.textContent = this.stats.totalQueries;
        this.weatherRequestsEl.textContent = this.stats.weatherRequests;
        this.aqiRequestsEl.textContent = this.stats.aqiRequests;
    }

    updateStatus(status, statusClass = '') {
        this.connectionStatus.textContent = status;
        this.connectionStatus.className = `status ${statusClass}`;
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the chat app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
});
