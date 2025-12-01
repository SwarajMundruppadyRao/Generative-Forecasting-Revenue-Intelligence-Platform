const API_BASE_URL = 'http://localhost:8000';

const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendButton = document.getElementById('sendButton');

// Example command mappings
const commandMappings = {
    'forecast': (store, dept) => `Forecast revenue for store ${store} department ${dept}`,
    'similar': (store) => `Which stores are similar to store ${store}?`,
    'insights': (store) => `Tell me about store ${store} performance`,
    'compare': (store) => `Compare LSTM and Transformer forecasts for store ${store}`,
    'trends': (dept) => `What are the trends for department ${dept}?`,
    'holiday': (store) => `How do holidays affect store ${store} sales?`
};

// Add message to chat
function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = isUser ? '👤' : '🤖';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    if (typeof content === 'string') {
        messageContent.innerHTML = formatMessage(content);
    } else {
        messageContent.appendChild(content);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Format message with markdown-like syntax
function formatMessage(text) {
    // Convert newlines to <br>
    text = text.replace(/\n/g, '<br>');
    
    // Convert **bold**
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Convert `code`
    text = text.replace(/`(.*?)`/g, '<code>$1</code>');
    
    return text;
}

// Create loading indicator
function createLoadingIndicator() {
    const loading = document.createElement('div');
    loading.className = 'loading';
    return loading;
}

// Parse user query and determine intent
async function processQuery(query) {
    const lowerQuery = query.toLowerCase();
    
    // Check for forecast intent
    if (lowerQuery.includes('forecast') || lowerQuery.includes('predict')) {
        const storeMatch = query.match(/store\s+(\d+)/i);
        const deptMatch = query.match(/department\s+(\d+)|dept\s+(\d+)/i);
        
        if (storeMatch && deptMatch) {
            const storeId = parseInt(storeMatch[1]);
            const deptId = parseInt(deptMatch[1] || deptMatch[2]);
            return await getForecast(storeId, deptId);
        }
    }
    
    // Check for similarity intent
    if (lowerQuery.includes('similar')) {
        const storeMatch = query.match(/store\s+(\d+)/i);
        if (storeMatch) {
            const storeId = parseInt(storeMatch[1]);
            return await getSimilarStores(storeId);
        }
    }
    
    // Check for comparison intent
    if (lowerQuery.includes('compare') && (lowerQuery.includes('lstm') || lowerQuery.includes('transformer'))) {
        const storeMatch = query.match(/store\s+(\d+)/i);
        const deptMatch = query.match(/department\s+(\d+)|dept\s+(\d+)/i);
        
        if (storeMatch) {
            const storeId = parseInt(storeMatch[1]);
            const deptId = deptMatch ? parseInt(deptMatch[1] || deptMatch[2]) : 1;
            return await compareModels(storeId, deptId);
        }
    }
    
    // Default: use RAG for general questions
    return await askRAG(query);
}

// Get forecast from API
async function getForecast(storeId, deptId) {
    try {
        const response = await fetch(`${API_BASE_URL}/forecast`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                store_id: storeId,
                dept_id: deptId,
                horizon: 4,
                model_type: 'lstm',
                natural_language_query: 'Explain the forecast'
            })
        });
        
        if (!response.ok) throw new Error('Forecast request failed');
        
        const data = await response.json();
        
        let result = `📊 **Forecast for Store ${storeId}, Department ${deptId}**\n\n`;
        result += `**Predictions:**\n`;
        data.predictions.forEach((pred, idx) => {
            result += `• Week ${idx + 1} (${data.forecast_dates[idx]}): $${pred.toFixed(2)}\n`;
        });
        
        if (data.explanation) {
            result += `\n**Analysis:**\n${data.explanation.substring(0, 500)}...`;
        }
        
        return result;
    } catch (error) {
        return `❌ Error getting forecast: ${error.message}`;
    }
}

// Get similar stores
async function getSimilarStores(storeId) {
    try {
        const response = await fetch(`${API_BASE_URL}/rag-answer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: `Which stores are similar to store ${storeId}?`,
                store_id: storeId
            })
        });
        
        if (!response.ok) throw new Error('RAG request failed');
        
        const data = await response.json();
        return `🔍 **Similar Stores to Store ${storeId}**\n\n${data.answer}`;
    } catch (error) {
        return `❌ Error finding similar stores: ${error.message}`;
    }
}

// Compare models
async function compareModels(storeId, deptId) {
    try {
        const [lstmResponse, transformerResponse] = await Promise.all([
            fetch(`${API_BASE_URL}/forecast`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    store_id: storeId,
                    dept_id: deptId,
                    horizon: 4,
                    model_type: 'lstm'
                })
            }),
            fetch(`${API_BASE_URL}/forecast`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    store_id: storeId,
                    dept_id: deptId,
                    horizon: 4,
                    model_type: 'transformer'
                })
            })
        ]);
        
        const lstmData = await lstmResponse.json();
        const transformerData = await transformerResponse.json();
        
        let result = `📈 **Model Comparison for Store ${storeId}, Department ${deptId}**\n\n`;
        result += `**LSTM Predictions:**\n`;
        lstmData.predictions.forEach((pred, idx) => {
            result += `• Week ${idx + 1}: $${pred.toFixed(2)}\n`;
        });
        
        result += `\n**Transformer Predictions:**\n`;
        transformerData.predictions.forEach((pred, idx) => {
            result += `• Week ${idx + 1}: $${pred.toFixed(2)}\n`;
        });
        
        const avgDiff = lstmData.predictions.reduce((sum, val, idx) => 
            sum + Math.abs(val - transformerData.predictions[idx]), 0) / lstmData.predictions.length;
        
        result += `\n**Average Difference:** $${avgDiff.toFixed(2)}`;
        
        return result;
    } catch (error) {
        return `❌ Error comparing models: ${error.message}`;
    }
}

// Ask RAG
async function askRAG(question) {
    try {
        const response = await fetch(`${API_BASE_URL}/rag-answer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        
        if (!response.ok) throw new Error('RAG request failed');
        
        const data = await response.json();
        return `💡 ${data.answer}`;
    } catch (error) {
        return `❌ Error: ${error.message}`;
    }
}

// Send message
async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;
    
    // Add user message
    addMessage(message, true);
    chatInput.value = '';
    
    // Add loading indicator
    const loadingDiv = createLoadingIndicator();
    addMessage(loadingDiv, false);
    
    // Process query
    const response = await processQuery(message);
    
    // Remove loading indicator
    chatMessages.removeChild(chatMessages.lastChild);
    
    // Add bot response
    addMessage(response, false);
}

// Event listeners
sendButton.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Example card click handlers
document.querySelectorAll('.example-card').forEach(card => {
    card.querySelector('.try-button').addEventListener('click', () => {
        const command = card.dataset.command;
        const store = card.dataset.store;
        const dept = card.dataset.dept;
        
        let query = '';
        if (command === 'forecast' && store && dept) {
            query = commandMappings[command](store, dept);
        } else if (command === 'similar' && store) {
            query = commandMappings[command](store);
        } else if (command === 'insights' && store) {
            query = commandMappings[command](store);
        } else if (command === 'compare' && store) {
            query = commandMappings[command](store);
        } else if (command === 'trends' && dept) {
            query = commandMappings[command](dept);
        } else if (command === 'holiday' && store) {
            query = commandMappings[command](store);
        }
        
        if (query) {
            chatInput.value = query;
            sendMessage();
        }
    });
});

// Smooth scroll for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});
